# %% General imports
import numpy as np
import torch

# DeepMoD stuff
from multitaskpinn import DeepMoD
from multitaskpinn.model.func_approx import NN
from multitaskpinn.model.library import Library1D
from multitaskpinn.model.constraint import LeastSquares
from multitaskpinn.model.sparse_estimators import Threshold
from multitaskpinn.training.sparsity_scheduler import TrainTestPeriodic
from multitaskpinn.training import train_SBL

from multitaskpinn.data import Dataset
from multitaskpinn.data.kdv import DoubleSoliton

# Settings
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# %% making data

x = np.linspace(-10, 10, 50)
t = np.linspace(0.0, 2.0, 20)
x_grid, t_grid = np.meshgrid(x, t, indexing='ij')

dataset = Dataset(DoubleSoliton, c=[5.0, 1.0], x0=[-5.0, -1.0])
X, y = dataset.create_dataset(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), n_samples=0, noise=0.20, random=True, normalize=True)
X, y = X.to(device), y.to(device)

# %% Running
n_runs = 5
for run in torch.arange(n_runs):
    sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=200, delta=1e-3) # in terms of write iterations
    network = NN(2, [30, 30, 30, 30, 30], 1)
    library = Library1D(poly_order=2, diff_order=3) # Library function
    estimator = Threshold(0.0) # Sparse estimator 
    constraint = LeastSquares() # How to constrain
    model = DeepMoD(network, library, estimator, constraint, 12).to(device) # Putting it all in the model
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.999), amsgrad=True, lr=1e-3) # Defining optimizer

    train_SBL(model, X, y, optimizer, sparsity_scheduler, exp_ID=f'SBL_run_{run}', split=0.8, write_iterations=50, max_iterations=15000, delta=1e-3, patience=200) 