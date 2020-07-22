




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
data = np.load('tests/keller_segel.npy', allow_pickle=True).item()
X = np.transpose((data['t'].flatten(), data['x'].flatten()))
y = np.transpose((data['u'].flatten(), data['v'].flatten()))
number_of_samples = 5000

idx = np.random.permutation(y.shape[0])
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)

# Configuring model
network = NN(2, [30, 30, 30, 30, 30], 2)  # Function approximator
library = Library1D(poly_order=2, diff_order=2) # Library function
estimator = Clustering() # Sparse estimator 
constraint = LeastSquares() # How to constrain
model = DeepMoD(network, library, estimator, constraint) # Putting it all in the model
model.s = torch.nn.Parameter(torch.zeros(2))

# Running model
sparsity_scheduler = Periodic(initial_epoch=12000, periodicity=100) # Defining when to apply sparsity
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.999), amsgrad=True) # Defining optimizer
train_optim(model, X_train, y_train, optimizer, sparsity_scheduler, max_iterations=10000, patience=500, delta=0.0001) # Running
print(model.s)