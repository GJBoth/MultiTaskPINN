import torch
from ..utils.logger import Logger
from .convergence import Convergence
from ..model.deepmod import DeepMoD


def train(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
          split: float = 0.8,
          exp_ID: str = None,
          log_dir: str = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """Trains the DeepMoD model. This function automatically splits the data set in a train and test set. 

    Args:
        model (DeepMoD):  A DeepMoD object.
        data (torch.Tensor):  Tensor of shape (n_samples x (n_spatial + 1)) containing the coordinates, first column should be the time coordinate.
        target (torch.Tensor): Tensor of shape (n_samples x n_features) containing the target data.
        optimizer ([type]):  Pytorch optimizer.
        sparsity_scheduler ([type]):  Decides when to update the sparsity mask.
        split (float, optional):  Fraction of the train set, by default 0.8.
        exp_ID (str, optional): Unique ID to identify tensorboard file. Not used if log_dir is given, see pytorch documentation.
        log_dir (str, optional): Directory where tensorboard file is written, by default None.
        max_iterations (int, optional): [description]. Max number of epochs , by default 10000.
        write_iterations (int, optional): [description]. Sets how often data is written to tensorboard and checks train loss , by default 25.
    """
    logger = Logger(exp_ID, log_dir)
    sparsity_scheduler.path = logger.log_dir # write checkpoint to same folder as tb output.

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)
    
    # Training
    convergence = Convergence(**convergence_kwargs)
    for iteration in torch.arange(0, max_iterations):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        loss = torch.sum(MSE + Reg) 

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            with torch.no_grad():
                prediction_test = model.func_approx(data_test)[0]
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
         
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating estimator coeffs but not setting mask
            logger(iteration, 
                   loss, MSE, Reg,
                   model.constraint_coeffs(sparse=True, scaled=True), 
                   model.constraint_coeffs(sparse=True, scaled=False),
                   model.estimator_coeffs(),
                   MSE_test=MSE_test)

            # ================== Sparsity update =============
            # Updating sparsity 
            update_sparsity = sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
            if update_sparsity: 
                model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)

            # ================= Checking convergence
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)))
            converged = convergence(iteration, l1_norm)
            if converged:
                break
    logger.close(model)


def train_SBL(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
          split = 0.8,
          exp_ID: str = None,
          log_dir: str = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """Trains the DeepMoD model. This function automatically splits the data set in a train and test set. 

    Args:
        model (DeepMoD):  A DeepMoD object.
        data (torch.Tensor):  Tensor of shape (n_samples x (n_spatial + 1)) containing the coordinates, first column should be the time coordinate.
        target (torch.Tensor): Tensor of shape (n_samples x n_features) containing the target data.
        optimizer ([type]):  Pytorch optimizer.
        sparsity_scheduler ([type]):  Decides when to update the sparsity mask.
        split (float, optional):  Fraction of the train set, by default 0.8.
        exp_ID (str, optional): Unique ID to identify tensorboard file. Not used if log_dir is given, see pytorch documentation.
        log_dir (str, optional): Directory where tensorboard file is written, by default None.
        max_iterations (int, optional): [description]. Max number of epochs , by default 10000.
        write_iterations (int, optional): [description]. Sets how often data is written to tensorboard and checks train loss , by default 25.
    """
    logger = Logger(exp_ID, log_dir)
    sparsity_scheduler.path = logger.log_dir # write checkpoint to same folder as tb output.
    
    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)

    n_samples = data_train.shape[0]
    model.t.data = -torch.var(target)
    model.b.data = -torch.var(target)
    threshold = 1e4
    # Training
    convergence = Convergence(**convergence_kwargs)
    for iteration in torch.arange(0, max_iterations):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        tau_ = torch.exp(model.t).clamp(max=1e8) #torch.min(torch.exp(model.t), torch.tensor(1e8, dtype=torch.float32))
        alpha_ = torch.exp(model.t).clamp(max=2*threshold) #torch.min(torch.exp(model.a), torch.tensor(1e8, dtype=torch.float32)) # we train the log of these things since they're very big
        beta_ = torch.exp(model.t).clamp(max=1e8) #torch.min(torch.exp(model.b), torch.tensor(1e8, dtype=torch.float32)) # we cap alpha and beta to prevent overflow

        p_reg, mn = SBL_loss(thetas[0], time_derivs[0], alpha_, beta_, threshold)
        p_MSE = tau_ * torch.sum((prediction - target_train)**2, dim=0) - n_samples * torch.log(tau_)
        loss = torch.sum(p_MSE + p_reg)
            
        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            with torch.no_grad():
                prediction_test = model.func_approx(data_test)[0]
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
         
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating estimator coeffs but not setting mask
            logger(iteration, 
                   loss, MSE, Reg,
                   model.constraint_coeffs(sparse=True, scaled=True), 
                   model.constraint_coeffs(sparse=True, scaled=False),
                   model.estimator_coeffs(),
                   MSE_test=MSE_test,
                   p_MSE=p_MSE,
                   p_reg=p_reg,
                   tau=tau_,
                   alpha=alpha_,
                   beta_=beta_,
                   mn=mn)

            # ================== Sparsity update =============
            # Updating sparsity 
            update_sparsity = sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
            #if update_sparsity: 
            #    model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)

            # ================= Checking convergence
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)))
            converged = convergence(iteration, l1_norm)
            if converged:
                break
    logger.close(model)
    

def SBL_loss(X, y, alpha_, beta_, threshold=1e4):
    n_samples = X.shape[0]
    mask = alpha_ < threshold
    A_inv = torch.inverse(torch.diag(alpha_[mask]) + beta_ * X[:, mask].T @ X[:, mask])
    mn = torch.zeros((alpha_.shape[0], 1)).to(X.device)
    mn[mask, :] = beta_ * A_inv @ X[:, mask].T @ y
    E = beta_ * torch.sum((y - X @ mn)**2) + (alpha_[:, None] * mn**2).sum()

    p_reg = E - (torch.logdet(A_inv) + n_samples * torch.log(beta_) + torch.sum(torch.log(alpha_))) # we use alpha and lambda since these are bounded
    return p_reg, mn
