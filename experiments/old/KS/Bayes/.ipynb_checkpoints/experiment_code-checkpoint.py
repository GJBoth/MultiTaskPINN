from multitaskpinn.utils.tensorboard import Tensorboard
from multitaskpinn.utils.output import progress
from multitaskpinn.model.deepmod import DeepMoD
from typing import Optional
import torch
import time       
import numpy as np
from torch.distributions.studentT import StudentT
from sklearn.linear_model import BayesianRidge


def train(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
          reg_weight,
          split: float = 0.8,
          log_dir: Optional[str] = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """Stops training when it reaches minimum MSE.

    Args:
        model (DeepMoD): [description]
        data (torch.Tensor): [description]
        target (torch.Tensor): [description]
        optimizer ([type]): [description]
        sparsity_scheduler ([type]): [description]
        log_dir (Optional[str], optional): [description]. Defaults to None.
        max_iterations (int, optional): [description]. Defaults to 10000.
    """
    start_time = time.time()
    board = Tensorboard(log_dir)  # initializing tb board

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)
    
    # Training
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        loss = torch.sum(MSE + reg_weight * Reg) 

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            prediction_test, coordinates = model.func_approx(data_test)
            time_derivs_test, thetas_test = model.library((prediction_test, coordinates))
            with torch.no_grad():
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
                Reg_test = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs_test, thetas_test, model.constraint_coeffs(scaled=False, sparse=True))])
                loss_test = torch.sum(MSE_test + reg_weight * Reg_test) 
            
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test, Reg_test=Reg_test, loss_test=loss_test)
            
            # ================== Sparsity update =============
            # Updating sparsity and or convergence
            if iteration % write_iterations == 0:
                sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
                    
                if sparsity_scheduler.apply_sparsity is True:
                    with torch.no_grad():
                        model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                        break
    board.close()


def train_bayes_MSE_optim(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
          split: float = 0.8,
          log_dir: Optional[str] = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """Stops training when it reaches minimum MSE.

    Args:
        model (DeepMoD): [description]
        data (torch.Tensor): [description]
        target (torch.Tensor): [description]
        optimizer ([type]): [description]
        sparsity_scheduler ([type]): [description]
        log_dir (Optional[str], optional): [description]. Defaults to None.
        max_iterations (int, optional): [description]. Defaults to 10000.
    """
    start_time = time.time()
    board = Tensorboard(log_dir)  # initializing tb board

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)
    
    # Training
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        loss = torch.sum(torch.exp(-model.s[:, 0]) * MSE + torch.sum(model.s[:, 0])) 

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            prediction_test, coordinates = model.func_approx(data_test)
            time_derivs_test, thetas_test = model.library((prediction_test, coordinates))
            with torch.no_grad():
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
                Reg_test = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs_test, thetas_test, model.constraint_coeffs(scaled=False, sparse=True))])
                loss_test = torch.sum(MSE_test + Reg_test) 
            
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test, Reg_test=Reg_test, loss_test=loss_test)
            
            # ================== Sparsity update =============
            # Updating sparsity and or convergence
            if iteration % write_iterations == 0:
                sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
                    
                if sparsity_scheduler.apply_sparsity is True:
                    with torch.no_grad():
                        model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                        break
    board.close()
    
    
    
def train_bayes_full_optim(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
          split: float = 0.8,
          log_dir: Optional[str] = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """Stops training when it reaches minimum MSE.

    Args:
        model (DeepMoD): [description]
        data (torch.Tensor): [description]
        target (torch.Tensor): [description]
        optimizer ([type]): [description]
        sparsity_scheduler ([type]): [description]
        log_dir (Optional[str], optional): [description]. Defaults to None.
        max_iterations (int, optional): [description]. Defaults to 10000.
    """
    start_time = time.time()
    board = Tensorboard(log_dir)  # initializing tb board

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)
    
    # Training
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)
        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        
        t = time_derivs[0]
        Theta = thetas[0]
        if iteration == 0:
            sk_reg = BayesianRidge(fit_intercept=False, compute_score=True, alpha_1=0, alpha_2=0, lambda_1=0, lambda_2=0)
            sk_reg.fit(thetas[0].cpu().detach().numpy(), time_derivs[0].cpu().detach().numpy())
            model.s.data[:, 0] = torch.log(1 / MSE.data)
            model.s.data[:, 1] = torch.log(torch.tensor(sk_reg.lambda_))
            model.s.data[:, 2] = torch.log(torch.tensor(sk_reg.alpha_))
            
            
        tau = torch.exp(model.s[:, 0])#torch.exp(-model.s[:, 0])
        alpha = torch.exp(model.s[:, 1])#torch.exp(-model.s[:, 1])#torch.exp(model.s[:, 1]) #1 / MSE[0].data
        beta = torch.exp(model.s[:, 2])#torch.exp(-model.s[:, 2])#torch.exp(model.s[:, 2]) #torch.tensor(1e-5).to(Theta.device)
        
        M = Theta.shape[1]
        N = Theta.shape[0]
    
        # Posterior std and mean
        A = torch.eye(M).to(Theta.device) * alpha + beta * Theta.T @ Theta  
        mn = beta * torch.inverse(A) @ Theta.T @ t

        loss_reg = -1/2 * (M * torch.log(alpha) 
         + N * torch.log(beta)
         - beta * (t - Theta @ mn).T @ (t - Theta @ mn) - alpha * mn.T @ mn 
         - torch.trace(torch.log(A))
         - N * np.log(2*np.pi))
                       
        
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        print(N / 2 * tau * MSE - N / 2 * torch.log(tau), loss_reg)
        loss = torch.sum((N / 2 * tau * MSE - N / 2 * torch.log(tau)) + loss_reg) 

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            prediction_test, coordinates = model.func_approx(data_test)
            time_derivs_test, thetas_test = model.library((prediction_test, coordinates))
            with torch.no_grad():
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
                Reg_test = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs_test, thetas_test, model.constraint_coeffs(scaled=False, sparse=True))])
                loss_test = torch.sum(MSE_test + Reg_test) 
            
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test, Reg_test=Reg_test, loss_test=loss_test, s=model.s)
            
            # ================== Sparsity update =============
            # Updating sparsity and or convergence
            if iteration % write_iterations == 0:
                sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
                    
                if sparsity_scheduler.apply_sparsity is True:
                    with torch.no_grad():
                        model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                        break
    board.close()