import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from lightning import LightningModule


@dataclass
class FlowConfig:
    """Configuration for conditional flow matching model with learnable encoder."""
    # Data dimensions
    n_radial_bins: int = 15  # Input: 15 radial bins
    n_spatial_positions: int = 900  # Spatial dimension per sample
    n_power_bins: int = 255  # Power spectrum output dimension
    n_cosmo_params: int = 0  # Optional cosmological parameters
    
    # Encoder architecture
    encoder_channels: list = None  # CNN channels
    encoder_latent_dim: int = 64  # Compressed representation size
    
    # Flow network architecture
    flow_hidden_dims: list = None
    time_embed_dim: int = 128
    dropout_rate: float = 0.1
    
    # Training
    sigma_min: float = 1e-4
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [32, 64, 128, 256]
        if self.flow_hidden_dims is None:
            self.flow_hidden_dims = [512, 512, 512, 256]


class kSZEncoder(nn.Module):
    """
    Flexible MLP encoder for kSZ data.
    
    Accepts arbitrary flattened input and builds network dynamically.
    Input shape: (batch, any_feature_dim)
    Output: (batch, encoder_latent_dim)
    """
    
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        self.network = None
        self.input_dim = None
        
    def _build_network(self, input_dim: int):
        """Build network dynamically based on input dimension."""
        self.input_dim = input_dim
        
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(self.config.dropout_rate)
        ])
        
        # Hidden layers
        layers.extend([
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(self.config.dropout_rate),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(self.config.dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        ])
        
        # Output layer
        layers.append(nn.Linear(128, self.config.encoder_latent_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, any_feature_dim) - flattened kSZ data
               Can be (batch, 13500) for full spatial or (batch, 15) for aggregated
        Returns:
            latent: (batch, encoder_latent_dim)
        """
        # Build network on first forward pass
        if self.network is None:
            self._build_network(x.shape[1])
            # Move to same device as input
            self.network = self.network.to(x.device)
        
        return self.network(x)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for flow time."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ConditionalVectorField(nn.Module):
    """
    Vector field for conditional flow matching.
    
    Predicts velocity conditioned on:
    - Encoded kSZ features (learned compression)
    - Optional cosmological parameters
    """
    
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        
        # Context embedding (encoded kSZ + optional cosmo params)
        context_dim = config.encoder_latent_dim + config.n_cosmo_params
        self.context_embed = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # Main network
        input_dim = config.n_power_bins + config.time_embed_dim + 256
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.flow_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, config.n_power_bins))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                ksz_latent: torch.Tensor,
                cosmo_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_t: (batch, n_power_bins) - current flow state
            t: (batch,) - flow time
            ksz_latent: (batch, encoder_latent_dim) - encoded kSZ features
            cosmo_params: Optional (batch, n_cosmo_params)
        Returns:
            velocity: (batch, n_power_bins)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Context embedding
        if cosmo_params is not None:
            context = torch.cat([ksz_latent, cosmo_params], dim=-1)
        else:
            context = ksz_latent
        context_emb = self.context_embed(context)
        
        # Concatenate and predict velocity
        net_input = torch.cat([x_t, t_emb, context_emb], dim=-1)
        velocity = self.network(net_input)
        
        return velocity


class ConditionalFlowMatcher:
    """
    End-to-end conditional flow matching with learnable kSZ encoder.
    """
    
    def __init__(self, config: FlowConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Initialize encoder and flow
        self.encoder = kSZEncoder(config).to(device)
        self.vector_field = ConditionalVectorField(config).to(device)
        
        self.optimizer = None
        
    def initialize_optimizer(self, lr: float = 1e-4, weight_decay: float = 1e-5):
        """Initialize optimizer for both encoder and flow."""
        params = list(self.encoder.parameters()) + list(self.vector_field.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    def compute_loss(
        self,
        ksz_spatial: torch.Tensor,
        power_spectra: torch.Tensor,
        cosmo_params: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditional flow matching loss.
        
        Args:
            ksz_spatial: (batch, n_radial_bins, n_spatial_positions)
            power_spectra: (batch, n_power_bins)
            cosmo_params: Optional (batch, n_cosmo_params)
        Returns:
            loss, ksz_latent for monitoring
        """
        batch_size = power_spectra.shape[0]
        
        # Encode kSZ spatial data
        ksz_latent = self.encoder(ksz_spatial)
        
        # Sample flow time
        t = torch.rand(batch_size, device=self.device)
        
        # Sample noise
        x_0 = torch.randn_like(power_spectra)
        
        # Optimal transport path
        t_expanded = t[:, None]
        x_t = t_expanded * power_spectra + (1 - t_expanded) * x_0
        
        # Target velocity
        u_t = power_spectra - x_0
        
        # Predicted velocity
        v_t = self.vector_field(x_t, t, ksz_latent, cosmo_params)
        
        # Flow matching loss
        loss = torch.mean((v_t - u_t) ** 2)
        
        return loss, ksz_latent
    
    def train_step(
        self,
        ksz_spatial: torch.Tensor,
        power_spectra: torch.Tensor,
        cosmo_params: Optional[torch.Tensor] = None
    ) -> Tuple[float, torch.Tensor]:
        """Single training step."""
        self.encoder.train()
        self.vector_field.train()
        self.optimizer.zero_grad()
        
        loss, ksz_latent = self.compute_loss(ksz_spatial, power_spectra, cosmo_params)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.vector_field.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        return loss.item(), ksz_latent.detach()
    
    @torch.no_grad()
    def sample(
        self,
        ksz_spatial: torch.Tensor,
        cosmo_params: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        n_steps: int = 100,
        method: str = 'euler'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate power spectrum samples.
        
        Args:
            ksz_spatial: (batch, n_radial_bins, n_spatial_positions) or 
                        (n_radial_bins, n_spatial_positions)
            cosmo_params: Optional (batch, n_cosmo_params)
            n_samples: Number of samples per input
            n_steps: ODE integration steps
            method: 'euler' or 'midpoint'
        Returns:
            samples: (batch * n_samples, n_power_bins)
            ksz_latent: (batch, encoder_latent_dim) - for inspection
        """
        self.encoder.eval()
        self.vector_field.eval()
        
        # Handle single input
        if ksz_spatial.ndim == 2:
            ksz_spatial = ksz_spatial.unsqueeze(0)
        
        batch_size = ksz_spatial.shape[0]
        
        # Encode kSZ data
        ksz_latent = self.encoder(ksz_spatial)
        
        # Replicate for multiple samples
        ksz_latent_rep = ksz_latent.repeat_interleave(n_samples, dim=0)
        if cosmo_params is not None:
            cosmo_params_rep = cosmo_params.repeat_interleave(n_samples, dim=0)
        else:
            cosmo_params_rep = None
        
        # Initialize from noise
        x = torch.randn(
            batch_size * n_samples,
            self.config.n_power_bins,
            device=self.device
        )
        
        # ODE integration
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.full((batch_size * n_samples,), step * dt, device=self.device)
            
            if method == 'euler':
                v = self.vector_field(x, t, ksz_latent_rep, cosmo_params_rep)
                x = x + dt * v
            elif method == 'midpoint':
                v = self.vector_field(x, t, ksz_latent_rep, cosmo_params_rep)
                x_mid = x + 0.5 * dt * v
                t_mid = t + 0.5 * dt
                v_mid = self.vector_field(x_mid, t_mid, ksz_latent_rep, cosmo_params_rep)
                x = x + dt * v_mid
        
        return x, ksz_latent
    
    def save(self, path: str):
        """Save checkpoint."""
        torch.save({
            'config': self.config,
            'encoder_state': self.encoder.state_dict(),
            'flow_state': self.vector_field.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None
        }, path)
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.vector_field.load_state_dict(checkpoint['flow_state'])
        if checkpoint['optimizer_state'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])


class PkFlow(LightningModule):
    """
    PyTorch Lightning wrapper for ConditionalFlowMatcher.
    
    Enables easy training with Lightning's features:
    - Automatic optimization
    - Learning rate scheduling
    - Logging and checkpointing
    - Multi-GPU training
    """
    
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize encoder and flow networks
        self.encoder = kSZEncoder(config)
        self.vector_field = ConditionalVectorField(config)
        
    def get_loss(
        self,
        ksz_spatial: torch.Tensor,
        power_spectra: torch.Tensor,
        cosmo_params: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditional flow matching loss.
        
        Args:
            ksz_spatial: (batch, n_radial_bins, n_spatial_positions)
            power_spectra: (batch, n_power_bins)
            cosmo_params: Optional (batch, n_cosmo_params)
        Returns:
            loss: scalar tensor
            ksz_latent: (batch, encoder_latent_dim) for monitoring
        """
        batch_size = power_spectra.shape[0]
        
        # Encode kSZ spatial data
        ksz_latent = self.encoder(ksz_spatial)
        
        # Sample flow time uniformly from [0, 1]
        t = torch.rand(batch_size, device=self.device)
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(power_spectra)
        
        # Optimal transport path: x_t = t * x_1 + (1-t) * x_0
        t_expanded = t[:, None]
        x_t = t_expanded * power_spectra + (1 - t_expanded) * x_0
        
        # Target velocity: u_t = x_1 - x_0
        u_t = power_spectra - x_0
        
        # Predicted velocity from the vector field
        v_t = self.vector_field(x_t, t, ksz_latent, cosmo_params)
        
        # Flow matching loss: MSE between predicted and target velocity
        loss = torch.mean((v_t - u_t) ** 2)
        
        return loss, ksz_latent
    
    def training_step(self, batch, batch_idx):
        """
        Training step for Lightning.
        
        Args:
            batch: Dictionary or tuple containing:
                - 'ksz_spatial': (batch, n_radial_bins, n_spatial_positions)
                - 'power_spectra': (batch, n_power_bins)
                - 'cosmo_params': Optional (batch, n_cosmo_params)
            batch_idx: Batch index
        Returns:
            loss: scalar tensor
        """
        # Unpack batch
        if isinstance(batch, dict):
            ksz_spatial = batch['ksz_spatial']
            power_spectra = batch['power_spectra']
            cosmo_params = batch.get('cosmo_params', None)
        else:
            # Assume tuple: (ksz_spatial, power_spectra, cosmo_params)
            if len(batch) == 3:
                ksz_spatial, power_spectra, cosmo_params = batch
            else:
                ksz_spatial, power_spectra = batch
                cosmo_params = None
        
        # Compute loss
        loss, ksz_latent = self.get_loss(ksz_spatial, power_spectra, cosmo_params)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('ksz_latent_mean', ksz_latent.mean(), on_step=False, on_epoch=True)
        self.log('ksz_latent_std', ksz_latent.std(), on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for Lightning.
        
        Args:
            batch: Same format as training_step
            batch_idx: Batch index
        Returns:
            loss: scalar tensor
        """
        # Unpack batch
        if isinstance(batch, dict):
            ksz_spatial = batch['ksz_spatial']
            power_spectra = batch['power_spectra']
            cosmo_params = batch.get('cosmo_params', None)
        else:
            if len(batch) == 3:
                ksz_spatial, power_spectra, cosmo_params = batch
            else:
                ksz_spatial, power_spectra = batch
                cosmo_params = None
        
        # Compute loss
        loss, ksz_latent = self.get_loss(ksz_spatial, power_spectra, cosmo_params)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ksz_latent_mean', ksz_latent.mean(), on_step=False, on_epoch=True)
        self.log('val_ksz_latent_std', ksz_latent.std(), on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer, scheduler, and monitoring config
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10, min_lr=1e-8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
    
    @torch.no_grad()
    def sample(
        self,
        ksz_spatial: torch.Tensor,
        cosmo_params: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        n_steps: int = 100,
        method: str = 'euler'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate power spectrum samples via ODE integration.
        
        Args:
            ksz_spatial: (batch, n_radial_bins, n_spatial_positions) or 
                        (n_radial_bins, n_spatial_positions)
            cosmo_params: Optional (batch, n_cosmo_params)
            n_samples: Number of samples per input
            n_steps: ODE integration steps
            method: 'euler' or 'midpoint'
        Returns:
            samples: (batch * n_samples, n_power_bins)
            ksz_latent: (batch, encoder_latent_dim)
        """
        self.eval()
        
        # Handle single input
        if ksz_spatial.ndim == 2:
            ksz_spatial = ksz_spatial.unsqueeze(0)
        
        batch_size = ksz_spatial.shape[0]
        
        # Encode kSZ data
        ksz_latent = self.encoder(ksz_spatial)
        
        # Replicate for multiple samples
        ksz_latent_rep = ksz_latent.repeat_interleave(n_samples, dim=0)
        if cosmo_params is not None:
            cosmo_params_rep = cosmo_params.repeat_interleave(n_samples, dim=0)
        else:
            cosmo_params_rep = None
        
        # Initialize from noise x_0 ~ N(0, I)
        x = torch.randn(
            batch_size * n_samples,
            self.config.n_power_bins,
            device=self.device
        )
        
        # ODE integration from t=0 to t=1
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.full((batch_size * n_samples,), step * dt, device=self.device)
            
            if method == 'euler':
                v = self.vector_field(x, t, ksz_latent_rep, cosmo_params_rep)
                x = x + dt * v
            elif method == 'midpoint':
                v = self.vector_field(x, t, ksz_latent_rep, cosmo_params_rep)
                x_mid = x + 0.5 * dt * v
                t_mid = t + 0.5 * dt
                v_mid = self.vector_field(x_mid, t_mid, ksz_latent_rep, cosmo_params_rep)
                x = x + dt * v_mid
        
        return x, ksz_latent


