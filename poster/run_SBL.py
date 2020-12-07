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

from deepymod.data import Dataset
from deepymod.data.burgers import BurgersDelta

device = "cpu"

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Making dataset
v = 0.25
A = 1.0

x = np.linspace(-3, 4, 40)
t = np.linspace(0.1, 5.0, 25)
x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
dataset = Dataset(BurgersDelta, v=v, A=A)
X, y = dataset.create_dataset(
    x_grid.reshape(-1, 1),
    t_grid.reshape(-1, 1),
    n_samples=0,
    noise=0.1,
    random=True,
    normalize=True,
)
X, y = X.to(device), y.to(device)

n_runs = 5
for run in np.arange(n_runs):
    network = NN(2, [30, 30, 30, 30, 30], 1)
    library = Library1D(poly_order=2, diff_order=3)  # Library function
    estimator = Threshold(0.1)  # Sparse estimator
    constraint = LeastSquares()  # Ridge(l=1e-3)# # How to constrain
    model = DeepMoD(network, library, estimator, constraint, 12).to(
        device
    )  # Putting it all in the model

    sparsity_scheduler = TrainTestPeriodic(
        periodicity=100, patience=1e8, delta=1e-6
    )  # in terms of write iterations
    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=2e-3
    )  # Defining optimizer

    train_SBL(
        model,
        X,
        y,
        optimizer,
        sparsity_scheduler,
        exp_ID=f"SBL_diff_run_{run}",
        split=0.8,
        write_iterations=100,
        max_iterations=50000,
        delta=1e-3,
        patience=1e8,
    )

