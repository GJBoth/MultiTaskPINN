# General imports
import numpy as np
import torch

# DeepMoD stuff
from multitaskpinn import DeepMoD
from multitaskpinn.model.func_approx import NN
from multitaskpinn.model.library import Library1D
from multitaskpinn.model.constraint import LeastSquares
from multitaskpinn.model.sparse_estimators import Threshold
from multitaskpinn.training import train, train_multitask
from multitaskpinn.training.sparsity_scheduler import Periodic

from phimal_utilities.data import Dataset
from phimal_utilities.data.burgers import BurgersDelta

#if torch.cuda.is_available():
#    device ='cuda'
#else:
#    device = 'cpu'
device = 'cpu'

# Settings for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Making dataset
v = 0.1
A = 1.0

x = np.linspace(-3, 4, 100)
t = np.linspace(0.5, 5.0, 50)
x_grid, t_grid = np.meshgrid(x, t, indexing='ij')
dataset = Dataset(BurgersDelta, v=v, A=A)

# Defining model
n_runs = 5

for run_idx in np.arange(n_runs):
    network = NN(2, [30, 30, 30, 30, 30], 1)
    library = Library1D(poly_order=2, diff_order=3) # Library function
    estimator = Threshold(0.1) # Sparse estimator 
    constraint = LeastSquares() # How to constrain
    model = DeepMoD(network, library, estimator, constraint).to(device) # Putting it all in the model
    
    sparsity_scheduler = Periodic(periodicity=25, initial_epoch=10000)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), amsgrad=True) # Defining optimizer
    
    X, y = dataset.create_dataset(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), n_samples=1000, noise=0.4, random=True, normalize=False)
    X, y = X.to(device), y.to(device)
    
    train_multitask(model, X, y, optimizer, sparsity_scheduler, log_dir=f'data_high_noise/multitask_run_{run_idx}/', write_iterations=25, max_iterations=5000, delta=0.00, patience=100) # Running
    