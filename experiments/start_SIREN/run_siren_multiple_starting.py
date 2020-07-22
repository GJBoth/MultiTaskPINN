# General imports
import numpy as np
import torch

# DeepMoD Components
from multitaskpinn import DeepMoD
from multitaskpinn.model.func_approx import NN, Siren
from multitaskpinn.model.library import Library1D
from multitaskpinn.model.constraint import LeastSquares
from multitaskpinn.model.sparse_estimators import Threshold
from multitaskpinn.training import train
from multitaskpinn.training.sparsity_scheduler import Periodic

# Data
from phimal_utilities.data import Dataset
from phimal_utilities.data.kdv import DoubleSoliton

# Cuda
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Making data
dataset = Dataset(DoubleSoliton, c=(5, 2), x0=(-3, -1))

x_sample = np.linspace(-7, 5, 50)
t_sample = np.linspace(0.0, 1.0, 40)
x_grid_sample, t_grid_sample = np.meshgrid(x_sample, t_sample, indexing='ij')

X_train, y_train = dataset.create_dataset(x_grid_sample.reshape(-1, 1), t_grid_sample.reshape(-1, 1), n_samples=0, noise=0.1, normalize=True, random=True)

# Setting up model
runs = 5
starting_points = np.arange(100, 701, 100)

for sparsity_start_epoch in starting_points:
    for run in np.arange(runs):
        # Configuring model
        network = Siren(2, [30, 30, 30, 30, 30], 1)  # Function approximator
        library = Library1D(poly_order=2, diff_order=3) # Library function
        estimator = Threshold(0.1) #Clustering() # Sparse estimator 
        constraint = LeastSquares() # How to constrain
        model = DeepMoD(network, library, estimator, constraint) # Putting it all in the model

        # Running model
        sparsity_scheduler = Periodic(initial_epoch=sparsity_start_epoch, periodicity=100) # Defining when to apply sparsity
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True) # Defining optimizer
        train(model, X_train, y_train, optimizer, sparsity_scheduler, log_dir=f'runs/SIREN_start/start_{sparsity_start_epoch}_run_{run}/', max_iterations=500) # Running