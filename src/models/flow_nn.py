import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from lightning import LightningModule

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False


@dataclass
class FlowConfig:
    """Configuration for conditional flow matching model with learnable encoder."""
    # Data dimensions
    n_radial_bins: int = 15  # Input: 15 radial bins
    n_spatial_positions: int = 900  # Spatial dimension per sample
    n_power_bins: int = 118  # Power spectrum output dimension
    n_cosmo_params: int = 0  # Optional cosmological parameters
    
    # Encoder architecture
    encoder_latent_dim: int = 64  # Compressed representation size
    
    # Flow network architecture
    flow_hidden_dims: list = None
    time_embed_dim: int = 128
    context_embed_dim: int = 256
    dropout_rate: float = 0.1
    
    # Flow matching parameters
    sigma: float = 0.0  # Stochastic flow control (0 = deterministic)
    reverse: bool = False  # Enable reverse-time flow

    smoothness_weight: float = 0.1  # Weight for smoothness regularization
    
    def __post_init__(self):
        if self.flow_hidden_dims is None:
            self.flow_hidden_dims = [512, 512, 512, 256]


class kSZEncoder(nn.Module):
    """
    Flexible MLP encoder for kSZ data.
    
    Accepts arbitrary flattened input and builds network dynamically.
    Input shape: (batch, any_feature_dim) - e.g., (batch, 13500)
    Output: (batch, encoder_latent_dim) - e.g., (batch, 64)
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
        
        # Input layer: input_dim -> 512
        layers.extend([
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(self.config.dropout_rate)
        ])
        
        # Hidden layers: 512 -> 512 -> 256 -> 128
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
        
        # Output layer: 128 -> encoder_latent_dim
        layers.append(nn.Linear(128, self.config.encoder_latent_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim) - flattened kSZ data (e.g., 13500)
        Returns:
            latent: (batch, encoder_latent_dim) - e.g., (batch, 64)
        """
        # Build network on first forward pass
        if self.network is None:
            self._build_network(x.shape[1])
            self.network = self.network.to(x.device)
        
        return self.network(x)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for flow time."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) - time values in [0, 1]
        Returns:
            emb: (batch, embed_dim) - sinusoidal embeddings
        """
        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class VelocityModel(nn.Module):
    """
    Velocity field for conditional flow matching with standard interface.
    
    Predicts velocity v_t conditioned on:
    - x: Current state (batch, n_power_bins=118)
    - t: Flow time (batch,)
    - z: Conditioning tuple (ksz_latent, cosmo_params)
    """
    
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        
        # Time embedding: (batch,) -> (batch, time_embed_dim=128)
        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        
        # Context embedding: (batch, encoder_latent_dim + n_cosmo_params) -> (batch, context_embed_dim=256)
        context_dim = config.encoder_latent_dim + config.n_cosmo_params
        self.context_embed = nn.Sequential(
            nn.Linear(context_dim, config.context_embed_dim),
            nn.SiLU(),
            nn.Linear(config.context_embed_dim, config.context_embed_dim)
        )
        
        # Main network: (batch, n_power_bins + time_embed_dim + context_embed_dim) -> (batch, n_power_bins)
        # Input: 118 + 128 + 256 = 502
        input_dim = config.n_power_bins + config.time_embed_dim + config.context_embed_dim
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers: 502 -> 512 -> 512 -> 512 -> 256
        for hidden_dim in config.flow_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer: 256 -> n_power_bins=118
        layers.append(nn.Linear(prev_dim, config.n_power_bins))
        
        self.network = nn.Sequential(*layers)
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        z: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Standard flow matching interface.
        
        Args:
            x: (batch, n_power_bins=118) - current state
            t: (batch,) - flow time in [0, 1]
            z: Conditioning tuple (ksz_latent, cosmo_params)
                - ksz_latent: (batch, encoder_latent_dim=64)
                - cosmo_params: (batch, n_cosmo_params) or None
        Returns:
            velocity: (batch, n_power_bins=118)
        """
        # Time embedding: (batch,) -> (batch, 128)
        t_emb = self.time_embed(t)
        
        # Unpack and embed conditioning
        if z is not None:
            ksz_latent, cosmo_params = z
            if cosmo_params is not None:
                # (batch, 64 + n_cosmo_params)
                context = torch.cat([ksz_latent, cosmo_params], dim=-1)
            else:
                # (batch, 64)
                context = ksz_latent
        else:
            # Unconditional fallback (shouldn't happen)
            context = torch.zeros(
                x.shape[0], 
                self.config.encoder_latent_dim,
                device=x.device, 
                dtype=x.dtype
            )
        
        # Context embedding: (batch, 64 + n_cosmo_params) -> (batch, 256)
        context_emb = self.context_embed(context)
        
        # Concatenate: (batch, 118 + 128 + 256) = (batch, 502)
        net_input = torch.cat([x, t_emb, context_emb], dim=-1)
        
        # Predict velocity: (batch, 502) -> (batch, 118)
        velocity = self.network(net_input)
        
        return velocity


class ConditionedVelocityModel(nn.Module):
    """
    Wrapper for ODE integration with conditioning and optional reverse.
    
    This wrapper adapts the velocity model to the ODE solver interface.
    """
    
    def __init__(
        self,
        velocity_model: nn.Module,
        z: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]],
        reverse: bool = False,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.z = z
        self.reverse = reverse

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        ODE function signature: dx/dt = f(t, x)
        
        Note: torchdiffeq expects (t, x) order.
        
        Args:
            t: Scalar or (batch,) time
            x: (batch, n_power_bins=118)
        Returns:
            velocity: (batch, 118)
        """
        # Broadcast scalar t to batch
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        
        velocity = self.velocity_model(x, t, self.z)
        return -velocity if self.reverse else velocity


class FlowMatching(nn.Module):
    """
    Flow matching module implementing conditional optimal transport.
    
    This is a standalone, reusable module that handles the flow matching
    logic independently of the Lightning training wrapper.
    """
    
    def __init__(
        self,
        velocity_model: nn.Module,
        sigma: float = 0.0,
        reverse: bool = False,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.sigma = sigma
        self.reverse = reverse

    def get_mu_t(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimal transport interpolation: μ_t = t*x1 + (1-t)*x0
        
        Args:
            x0: (batch, n_power_bins=118) - initial noise
            x1: (batch, 118) - target data
            t: (batch, 1) - time broadcasted
        Returns:
            mu_t: (batch, 118)
        """
        return t * x1 + (1 - t) * x0

    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Stochastic variance schedule: σ_t = sqrt(2*t*(1-t))
        
        Args:
            t: (batch, 1)
        Returns:
            sigma_t: (batch, 1)
        """
        return torch.sqrt(2 * t * (1 - t))

    def sample_xt(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        t: torch.Tensor, 
        eps: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Sample from conditional distribution at time t.
        
        Args:
            x0: (batch, 118) - initial noise
            x1: (batch, 118) - target data
            t: (batch,) - time values
            eps: (batch, 118) - optional noise for stochastic flow
        Returns:
            x_t: (batch, 118) - interpolated sample
        """
        # Broadcast t to match data dimensions: (batch,) -> (batch, 1)
        t_broadcast = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        
        # Compute mean: (batch, 118)
        mu_t = self.get_mu_t(x0, x1, t_broadcast)
        
        # Add stochastic noise if sigma > 0
        if self.sigma != 0.0 and eps is not None:
            sigma_t = self.get_sigma_t(t_broadcast)
            return mu_t + sigma_t * eps
        return mu_t

    def compute_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        z: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            x0: (batch, 118) - initial noise
            x1: (batch, 118) - target data
            z: Conditioning tuple (ksz_latent, cosmo_params)
            t: Optional (batch,) - time samples
        Returns:
            loss: Scalar MSE loss
        """
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        
        # Sample noise for stochastic flow
        eps = torch.randn_like(x0) if self.sigma != 0.0 else None
        
        # Sample x_t from conditional distribution: (batch, 118)
        xt = self.sample_xt(x0, x1, t, eps)
        
        # Target velocity (optimal transport): (batch, 118)
        ut = x1 - x0
        
        # Predicted velocity: (batch, 118)
        vt = self.velocity_model(xt, t, z)
        
        # Flow matching loss
        return torch.mean((vt - ut) ** 2)


class PkFlow(LightningModule):
    """
    PyTorch Lightning wrapper for conditional flow matching.
    
    Production-quality implementation with:
    - Modular flow matching architecture
    - Adaptive ODE solvers (when torchdiffeq available)
    - Multi-GPU support
    - Comprehensive logging
    - Output smoothness regularization
    
    Data flow:
    1. kSZ input (batch, 15, 900) -> flatten -> (batch, 13500)
    2. Encoder: (batch, 13500) -> (batch, 64)
    3. Flow: noise (batch, 118) -> power spectrum (batch, 118)
    """
    
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.smoothness_weight = config.smoothness_weight
        
        # Initialize components
        self.encoder = kSZEncoder(config)
        self.velocity_model = VelocityModel(config)
        
        # Flow matching module
        self.flow_matcher = FlowMatching(
            velocity_model=self.velocity_model,
            sigma=config.sigma,
            reverse=config.reverse
        )
    
    def _unpack_batch(
        self, 
        batch
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Centralized batch unpacking with support for dict/tuple formats."""
        if isinstance(batch, dict):
            return (
                batch['ksz_spatial'],
                batch['power_spectra'],
                batch.get('cosmo_params')
            )
        elif len(batch) == 3:
            return batch
        else:
            return (*batch, None)
    
    def compute_smoothness_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness loss on power spectrum predictions.
        Penalizes large differences between adjacent k-bins.
        
        Args:
            predictions: (batch, n_power_bins) - predicted power spectra
        Returns:
            loss: Scalar smoothness loss
        """
        # Finite differences: |P(k_i+1) - P(k_i)|
        diff = predictions[:, 1:] - predictions[:, :-1]
        
        # L2 penalty on differences (encourages smooth curves)
        smoothness_loss = torch.mean(diff ** 2)
        
        return smoothness_loss
        
    def get_loss(
        self,
        ksz_spatial: torch.Tensor,
        power_spectra: torch.Tensor,
        cosmo_params: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute conditional flow matching loss with smoothness regularization.
        
        Args:
            ksz_spatial: (batch, feature_dim) - flattened kSZ (batch, 13500)
            power_spectra: (batch, 118) - target power spectra
            cosmo_params: Optional (batch, n_cosmo_params) - cosmological parameters
        Returns:
            total_loss: Scalar tensor (flow loss + smoothness)
            ksz_latent: (batch, 64) - encoded kSZ for monitoring
            smoothness_loss: Scalar smoothness loss for logging
        """
        # Encode kSZ data: (batch, 13500) -> (batch, 64)
        ksz_latent = self.encoder(ksz_spatial)
        
        # Package conditioning
        z = (ksz_latent, cosmo_params)
        
        # Sample initial noise: (batch, 118)
        x0 = torch.randn_like(power_spectra)
        
        # Compute flow matching loss
        flow_loss = self.flow_matcher.compute_loss(x0, power_spectra, z)
        
        # Compute smoothness loss on target (encourages smooth velocity field)
        smoothness_loss = self.compute_smoothness_loss(power_spectra)
        
        # Total loss
        total_loss = flow_loss + self.smoothness_weight * smoothness_loss
        
        return total_loss, ksz_latent, smoothness_loss
    
    def training_step(self, batch, batch_idx):
        """Training step with automatic optimization."""
        ksz_spatial, power_spectra, cosmo_params = self._unpack_batch(batch)
        
        loss, ksz_latent, smoothness_loss = self.get_loss(ksz_spatial, power_spectra, cosmo_params)
        
        # Log metrics with multi-GPU sync
        self.log('train_loss', loss, on_step=True, on_epoch=True, 
                 prog_bar=True, sync_dist=True)
        self.log('train_smoothness_loss', smoothness_loss, on_step=False, on_epoch=True, 
                 sync_dist=True)
        self.log('ksz_latent_mean', ksz_latent.mean(), 
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('ksz_latent_std', ksz_latent.std(), 
                 on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        ksz_spatial, power_spectra, cosmo_params = self._unpack_batch(batch)
        
        loss, ksz_latent, smoothness_loss = self.get_loss(ksz_spatial, power_spectra, cosmo_params)
        
        # Log validation metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, 
                 prog_bar=True, sync_dist=True)
        self.log('val_smoothness_loss', smoothness_loss, on_step=False, on_epoch=True, 
                 sync_dist=True)
        self.log('val_ksz_latent_mean', ksz_latent.mean(), 
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_ksz_latent_std', ksz_latent.std(), 
                 on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer with learning rate scheduling."""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
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
        solver: str = 'euler',
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate power spectrum samples via ODE integration.
        
        Args:
            ksz_spatial: (batch, 13500) - flattened kSZ data
            cosmo_params: Optional (batch, n_cosmo_params)
            n_samples: Number of samples per input
            n_steps: Number of integration steps
            solver: ODE solver ('dopri5', 'rk4', 'euler', 'midpoint')
            atol: Absolute tolerance for adaptive solvers
            rtol: Relative tolerance for adaptive solvers
        Returns:
            samples: (batch * n_samples, 118) - generated power spectra
            ksz_latent: (batch, 64) - encoded conditioning
        """
        self.eval()
        
        # Handle single input
        if ksz_spatial.ndim == 1:
            ksz_spatial = ksz_spatial.unsqueeze(0)
        
        batch_size = ksz_spatial.shape[0]
        
        # Encode kSZ data: (batch, 13500) -> (batch, 64)
        ksz_latent = self.encoder(ksz_spatial)
        
        # Replicate for multiple samples
        ksz_latent_rep = ksz_latent.repeat_interleave(n_samples, dim=0)
        cosmo_params_rep = (
            cosmo_params.repeat_interleave(n_samples, dim=0) 
            if cosmo_params is not None else None
        )
        
        # Initial noise: (batch * n_samples, 118)
        x0 = torch.randn(
            batch_size * n_samples,
            self.config.n_power_bins,
            device=self.device,
            dtype=ksz_spatial.dtype
        )
        
        # Choose solver
        if solver in ['dopri5', 'rk4', 'adaptive_heun'] and TORCHDIFFEQ_AVAILABLE:
            samples = self._sample_adaptive(
                ksz_latent_rep, cosmo_params_rep, x0,
                n_steps, solver, atol, rtol
            )
        elif solver in ['euler', 'midpoint']:
            samples = self._sample_fixed(
                ksz_latent_rep, cosmo_params_rep, x0, n_steps, solver
            )
        else:
            if solver in ['dopri5', 'rk4', 'adaptive_heun']:
                print(f"Warning: {solver} requires torchdiffeq. Falling back to Euler.")
                samples = self._sample_fixed(
                    ksz_latent_rep, cosmo_params_rep, x0, n_steps, 'euler'
                )
            else:
                raise ValueError(
                    f"Unknown solver: {solver}. "
                    f"Choose from ['dopri5', 'rk4', 'euler', 'midpoint', 'adaptive_heun']"
                )
        
        return samples, ksz_latent
    
    def _sample_adaptive(
        self,
        ksz_latent: torch.Tensor,
        cosmo_params: Optional[torch.Tensor],
        x0: torch.Tensor,
        n_steps: int,
        solver: str,
        atol: float,
        rtol: float,
    ) -> torch.Tensor:
        """
        Sample using adaptive ODE solver (torchdiffeq).
        
        Args:
            ksz_latent: (batch * n_samples, 64)
            cosmo_params: Optional (batch * n_samples, n_cosmo_params)
            x0: (batch * n_samples, 118)
            n_steps: Number of time points
            solver: ODE solver method
            atol: Absolute tolerance
            rtol: Relative tolerance
        Returns:
            samples: (batch * n_samples, 118)
        """
        # Package conditioning
        z = (ksz_latent, cosmo_params)
        
        # Create conditioned velocity model for ODE
        conditioned_velocity = ConditionedVelocityModel(
            self.velocity_model, z, self.config.reverse
        )
        
        # Solve ODE with adaptive solver
        t_span = torch.linspace(0, 1, n_steps, device=x0.device, dtype=x0.dtype)
        trajectory = odeint(
            conditioned_velocity,
            x0,
            t_span,
            method=solver,
            atol=atol,
            rtol=rtol
        )
        
        return trajectory[-1]
    
    def _sample_fixed(
        self,
        ksz_latent: torch.Tensor,
        cosmo_params: Optional[torch.Tensor],
        x0: torch.Tensor,
        n_steps: int,
        method: str,
    ) -> torch.Tensor:
        """
        Sample using fixed-step manual integration.
        
        Args:
            ksz_latent: (batch * n_samples, 64)
            cosmo_params: Optional (batch * n_samples, n_cosmo_params)
            x0: (batch * n_samples, 118)
            n_steps: Number of integration steps
            method: 'euler' or 'midpoint'
        Returns:
            samples: (batch * n_samples, 118)
        """
        z = (ksz_latent, cosmo_params)
        x = x0
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.full(
                (x.shape[0],), 
                step * dt, 
                device=x.device, 
                dtype=x.dtype
            )
            
            if method == 'euler':
                v = self.velocity_model(x, t, z)
                if self.config.reverse:
                    v = -v
                x = x + dt * v
                
            elif method == 'midpoint':
                v = self.velocity_model(x, t, z)
                if self.config.reverse:
                    v = -v
                x_mid = x + 0.5 * dt * v
                
                t_mid = t + 0.5 * dt
                v_mid = self.velocity_model(x_mid, t_mid, z)
                if self.config.reverse:
                    v_mid = -v_mid
                x = x + dt * v_mid
        
        return x


