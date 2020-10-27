from multitaskpinn.utils.tensorboard import Tensorboard
from multitaskpinn.utils.output import progress
from multitaskpinn.model.deepmod import DeepMoD
from typing import Optional
import torch
import time       
import numpy as np
from torch.distributions.studentT import StudentT

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



def train_mt(model: DeepMoD,
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
    
    cutoff = torch.tensor(15.).to(target.device)
    # Training
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        s_capped = torch.min(torch.max(model.s, -cutoff), cutoff)
        loss = torch.sum(2 * torch.exp(-s_capped[:, 0]) * MSE + torch.exp(-s_capped[:, 1]) * Reg + torch.sum(s_capped)) 

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
                loss_test = torch.sum(torch.exp(-s_capped[:, 0]) * MSE_test + torch.exp(-s_capped[:, 1]) * Reg_test + torch.sum(s_capped)) 
            
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


def train_gradnorm(model: DeepMoD,
                    data: torch.Tensor,
                    target: torch.Tensor,
                    optimizer,
                    sparsity_scheduler,
                    alpha,
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
        Reg = torch.cat([torch.mean((dt - theta @ coeff_vector)**2, dim=0)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        task_loss = (torch.exp(model.weights) * torch.stack((MSE, Reg), axis=1)).flatten() # weighted losses
        loss = torch.sum(task_loss)

        if iteration == 0: # Getting initial loss
            ini_loss = task_loss.data
        if torch.any(task_loss.data > ini_loss):
            ini_loss[task_loss.data > ini_loss] = task_loss.data[task_loss.data > ini_loss]

        # Getting original grads
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        model.weights.grad.data = model.weights.grad.data * 0.0 # setting weight grads to zero

        # Getting Grads to normalize
        G = torch.tensor([torch.norm(torch.autograd.grad(loss_i, list(model.parameters())[-2], retain_graph=True, create_graph=True)[0], 2) for loss_i in task_loss]).to(data.device)
        G_mean = torch.mean(G)  

        # Calculating relative losses
        rel_loss = task_loss / ini_loss
        inv_train_rate = rel_loss / torch.mean(rel_loss)

        # Calculating grad norm loss
        grad_norm_loss = torch.sum(torch.abs(G - G_mean * inv_train_rate ** alpha))

        # Setting grads
        model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]
    
        # do a step with the optimizer
        optimizer.step()
        
        # renormalize
        normalize_coeff = task_loss.shape[0] / torch.sum(model.weights)
        model.weights.data = torch.log(torch.exp(model.weights.data) * normalize_coeff)

        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            prediction_test, coordinates = model.func_approx(data_test)
            time_derivs_test, thetas_test = model.library((prediction_test, coordinates))
            with torch.no_grad():
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
                Reg_test = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs_test, thetas_test, model.constraint_coeffs(scaled=False, sparse=True))])
                loss_test = model.weights @ torch.stack((MSE, Reg), axis=0)
            
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test, Reg_test=Reg_test, loss_test=loss_test, w=model.weights)
            
            # ================== Sparsity update =============
            # Updating sparsity and or convergence
            if iteration % write_iterations == 0:
                sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
                    
                if sparsity_scheduler.apply_sparsity is True:
                    with torch.no_grad():
                        model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                        break
    board.close()


def train_scaled(model: DeepMoD,
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

        loss = torch.sum(MSE + torch.exp(- MSE.data / Reg.data) * Reg) 

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
                loss_test = torch.sum( MSE_test + Reg_test) 
            
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

def train_wscaled(model: DeepMoD,
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
        with torch.no_grad():
            w_Reg = 1 / torch.sum(MSE.data)
            w_MSE = 1 - w_Reg
        
        loss = torch.sum(w_MSE * MSE + w_Reg * Reg) 

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
                loss_test = torch.sum(w_MSE * MSE_test + w_Reg * Reg_test) 
            
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test, Reg_test=Reg_test, loss_test=loss_test, w_MSE=w_MSE, w_Reg=w_Reg)
            
            # ================== Sparsity update =============
            # Updating sparsity and or convergence
            if iteration % write_iterations == 0:
                sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
                    
                if sparsity_scheduler.apply_sparsity is True:
                    with torch.no_grad():
                        model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                        break
    board.close()
    
    
def train_bayes(model: DeepMoD,
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
        
        v = torch.tensor(10.).to(data.device)
        s0 = torch.tensor(1e-2).to(data.device)
        n = n_train
        v_MSE = v + n
        s0_MSE = (v * s0 + torch.sum((target_train - prediction)**2)) / (v + n)
        t = StudentT(v_MSE, loc=prediction, scale=s0_MSE)
        loss = -torch.mean(t.log_prob(target_train)) 

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
    
    
    
def train_LL(model: DeepMoD,
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
         
        s_MSE = MSE
        s_Reg = (MSE * Reg) / (MSE + Reg)
        loss = 1/s_MSE * MSE + 1/s_MSE * Reg + torch.log(s_MSE * s_MSE)

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