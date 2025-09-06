import torch
from torch import Tensor

from sbi.analysis import pairplot
from sbi.inference import NPE, FMPE, SNPE, VIPosterior, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from GP_dataloader import *

import argparse

# Better argument parsing
parser = argparse.ArgumentParser(description='Run NPE or FMPE inference')
parser.add_argument('--method', '-m', 
                   choices=['NPE', 'FMPE', 'SNPE'], 
                   default='NPE',
                   help='Inference method to use (default: NPE)')
parser.add_argument('--num_sims', '-n', 
                   type=int, 
                   default=1000,
                   help='Number of simulations (default: 1000)')
parser.add_argument('--sim_id', '-s',
                   type=int,
                   default=777,
                   help='Simulation ID for observation (default: 777)')
parser.add_argument('--num_rounds', '-r',
                   type=int,
                   default=3,
                   help='Number of rounds (default: 3)')

args = parser.parse_args()
inference_method = args.method
num_sims = args.num_sims
sim_id = args.sim_id
num_rounds = args.num_rounds
VariationalInference = True

param_names, fiducial_values, maxdiff, minVal, maxVal = getParamsFiducial()


prior = BoxUniform(low=Tensor(minVal), high=Tensor(maxVal))
simulator = lambda x: getProfilesParamsTensor(x, filterType='CAP', ptype='gas')

r_bins, x_o = getProfiles([sim_id], filterType='CAP', ptype='gas')
theta_true = torch.tensor(getParams([sim_id]), dtype=torch.float32)


# Set up the inference method
if inference_method == "FMPE":
    inference = FMPE(prior=prior)
elif inference_method == "NPE":
    inference = NPE(prior=prior)
elif inference_method == "SNPE":
    inference = SNPE(prior=prior)

posteriors = []
proposal = prior
for r in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=500)
    inference = inference.append_simulations(theta, x)
    final_round = r == num_rounds - 1
    density_estimator = inference.train(final_round=final_round)
    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)

if VariationalInference:
    vi_posterior = VIPosterior(posterior, num_components=1)

    samples = vi_posterior.train(
        num_steps=5000,  # optimization iterations
        show_progress_bars=True,
        x=x_o,  # condition on the observed data
    )
else:
    samples = posterior.sample((5000,), x=x_o)
    
pairplot(samples,
         points=theta_true,
         figsize=(20, 20),
         labels=param_names)