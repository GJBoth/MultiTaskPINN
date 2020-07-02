# General imports
import numpy as np
import torch

# DeepMoD stuff
from multitaskpinn.model.deepmod import DeepMoD
from multitaskpinn.model.func_approx import NN
from multitaskpinn.model.library import Library1D
from multitaskpinn.model.constraint import LeastSquares
from multitaskpinn.model.sparse_estimators import Clustering, Threshold
from multitaskpinn.training import train_optim
from multitaskpinn.training.sparsity_scheduler import Periodic
from phimal_utilities.data import Dataset
from phimal_utilities.data.burgers import BurgersDelta

# Setting cuda
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Making data
v = 0.1
A = 1.0
x = np.linspace(-3, 4, 100)
t = np.linspace(0.5, 5.0, 50)

x_grid, t_grid = np.meshgrid(x, t, indexing='ij')
dataset = Dataset(BurgersDelta, v=v, A=A)
X_train, y_train = dataset.create_dataset(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), n_samples=1000, noise=0.1)

# Configuring model
network = NN(2, [30, 30, 30, 30, 30], 1)  # Function approximator
library = Library1D(poly_order=2, diff_order=3) # Library function
estimator = Clustering() # Sparse estimator 
constraint = LeastSquares() # How to constrain
model = DeepMoD(network, library, estimator, constraint) # Putting it all in the model
model.s = torch.nn.Parameter(torch.zeros(2))

# Running model
sparsity_scheduler = Periodic(initial_epoch=12000, periodicity=100) # Defining when to apply sparsity
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.999), amsgrad=True) # Defining optimizer
train_optim(model, X_train, y_train, optimizer, sparsity_scheduler, max_iterations=10000, patience=500, delta=0.0001) # Running
print(model.s)