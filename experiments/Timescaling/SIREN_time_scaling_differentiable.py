# General imports
import numpy as np
import torch

# DeepMoD Components
from multitaskpinn import DeepMoD
from multitaskpinn.model.func_approx import Siren
from multitaskpinn.model.library import Library1D
from multitaskpinn.model.constraint import LeastSquares
from multitaskpinn.model.sparse_estimators import Threshold
from multitaskpinn.training import train_classic, train
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

# Making training set
dataset = Dataset(DoubleSoliton, c=(5, 2), x0=(-3, -1))

x_sample = np.linspace(-7, 5, 50)
t_sample = np.linspace(0.0, 1.0, 40)
x_grid_sample, t_grid_sample = np.meshgrid(x_sample, t_sample, indexing='ij')
X_train, y_train = dataset.create_dataset(x_grid_sample.reshape(-1, 1), t_grid_sample.reshape(-1, 1), n_samples=0, noise=0.1, normalize=True, random=True)

# Configuring model
network = Siren(2, [30, 30, 30, 30, 30], 1)  # Function approximator
library = Library1D(poly_order=2, diff_order=3) # Library function
estimator = Threshold(0.1) #Clustering() # Sparse estimator 
constraint = LeastSquares() # How to constrain
model = DeepMoD(network, library, estimator, constraint) # Putting it all in the model

# Running model
sparsity_scheduler = Periodic(initial_epoch=5000, periodicity=100) # Defining when to apply sparsity
optimizer = torch.optim.Adam(model.parameters(), betas=(0.999, 0.999), lr=0.00025, amsgrad=True) # Defining optimizer

train(model, X_train, y_train, optimizer, sparsity_scheduler, delta=0.0, log_dir='runs/dt_differentiable/', max_iterations=2000) # Running