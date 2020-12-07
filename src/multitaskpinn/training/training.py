import torch
from ..utils.logger import Logger
from .convergence import Convergence
from ..model.deepmod import DeepMoD
import numpy as np


def train(
    model: DeepMoD,
    data: torch.Tensor,
    target: torch.Tensor,
    optimizer,
    sparsity_scheduler,
    split: float = 0.8,
    exp_ID: str = None,
    log_dir: str = None,
    max_iterations: int = 10000,
    write_iterations: int = 25,
    **convergence_kwargs
) -> None:
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
    sparsity_scheduler.path = (
        logger.log_dir
    )  # write checkpoint to same folder as tb output.

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

        MSE = torch.mean((prediction - target_train) ** 2, dim=0)  # loss per output
        Reg = torch.stack(
            [
                torch.mean((dt - theta @ coeff_vector) ** 2)
                for dt, theta, coeff_vector in zip(
                    time_derivs,
                    thetas,
                    model.constraint_coeffs(scaled=False, sparse=True),
                )
            ]
        )
        loss = torch.sum(MSE + Reg)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            with torch.no_grad():
                prediction_test = model.func_approx(data_test)[0]
                MSE_test = torch.mean(
                    (prediction_test - target_test) ** 2, dim=0
                )  # loss per output

            # ====================== Logging =======================
            _ = model.sparse_estimator(
                thetas, time_derivs
            )  # calculating estimator coeffs but not setting mask
            logger(
                iteration,
                loss,
                MSE,
                Reg,
                model.constraint_coeffs(sparse=True, scaled=True),
                model.constraint_coeffs(sparse=True, scaled=False),
                model.estimator_coeffs(),
                MSE_test=MSE_test,
            )

            # ================== Sparsity update =============
            # Updating sparsity
            update_sparsity = sparsity_scheduler(
                iteration, torch.sum(MSE_test), model, optimizer
            )
            if update_sparsity:
                model.constraint.sparsity_masks = model.sparse_estimator(
                    thetas, time_derivs
                )

            # ================= Checking convergence
            l1_norm = torch.sum(
                torch.abs(
                    torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)
                )
            )
            converged = convergence(iteration, l1_norm)
            if converged:
                break
    logger.close(model)


def train_multitask(
    model: DeepMoD,
    data: torch.Tensor,
    target: torch.Tensor,
    optimizer,
    sparsity_scheduler,
    split=0.8,
    exp_ID: str = None,
    log_dir: str = None,
    max_iterations: int = 10000,
    write_iterations: int = 25,
    **convergence_kwargs
) -> None:
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
    sparsity_scheduler.path = (
        logger.log_dir
    )  # write checkpoint to same folder as tb output.

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)

    n_samples = data_train.shape[0]
    model.t.data = -torch.var(target)
    model.b.data = -torch.var(target)

    # Training
    convergence = Convergence(**convergence_kwargs)
    for iteration in torch.arange(0, max_iterations):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        # we train the log of these things since they're very big
        # we cap alpha and beta to prevent overflow
        tau_ = torch.exp(model.t).clamp(max=1e8)
        beta_ = torch.exp(model.b).clamp(max=1e8)

        MSE = torch.mean((prediction - target_train) ** 2, dim=0)  # loss per output
        Reg = torch.stack(
            [
                torch.mean((dt - theta @ coeff_vector) ** 2)
                for dt, theta, coeff_vector in zip(
                    time_derivs,
                    thetas,
                    model.constraint_coeffs(scaled=False, sparse=True),
                )
            ]
        )

        p_MSE = n_samples * (tau_ * MSE - torch.log(tau_))
        p_reg = n_samples * (beta_ * Reg - torch.log(beta_))
        loss = torch.sum(p_MSE + p_reg)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            with torch.no_grad():
                prediction_test = model.func_approx(data_test)[0]
                MSE_test = torch.mean(
                    (prediction_test - target_test) ** 2, dim=0
                )  # loss per output

            # ====================== Logging =======================
            _ = model.sparse_estimator(
                thetas, time_derivs
            )  # calculating estimator coeffs but not setting mask
            logger(
                iteration,
                loss,
                MSE,
                Reg,
                model.constraint_coeffs(sparse=True, scaled=True),
                model.constraint_coeffs(sparse=True, scaled=False),
                model.estimator_coeffs(),
                MSE_test=MSE_test,
                p_MSE=p_MSE,
                p_reg=p_reg,
                tau=tau_,
                beta_=beta_,
            )

            # ================== Sparsity update =============
            # Updating sparsity
            update_sparsity = sparsity_scheduler(
                iteration, torch.sum(MSE_test), model, optimizer
            )
            if update_sparsity:
                model.constraint.sparsity_masks = model.sparse_estimator(
                    thetas, time_derivs
                )

            # ================= Checking convergence
            l1_norm = torch.sum(
                torch.abs(
                    torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)
                )
            )
            converged = convergence(iteration, l1_norm)
            if converged:
                break
    logger.close(model)


def train_bayes_type_II(
    model: DeepMoD,
    data: torch.Tensor,
    target: torch.Tensor,
    optimizer,
    sparsity_scheduler,
    split=0.8,
    exp_ID: str = None,
    log_dir: str = None,
    max_iterations: int = 10000,
    write_iterations: int = 25,
    **convergence_kwargs
) -> None:
    """Trains the DeepMoD model. This function automatically splits the data set in a train and test set. 

    Args:
        model (DeepMoD):  A DeepMoD object.
        data (torch.Tensor):  Tensor of shape (n_samples x (n_spatial + 1)) containing the coordinates, first column should be the time coordinate.
        target (torch.Tensor): Tensor of shape (n_samples x n_features) containing the target data.
        optimizer ([type]):  Pytorch optimizer.
        sparsity_scheduler ([type]):  Decides when to update the sparsity mask.
        split (float, optional):  Fraction of the train set, by default 0.8.
        exp_ID (str, optional): Unique ID to identify tensorboard file. Not used if log_dir is given, see pytorch documentation.
        log_dir (str, optional): Directory where tensorboard file is writtenp by default None.
        max_iterations (int, optional): [description]. Max number of epochs , by default 10000.
        write_iterations (int, optional): [description]. Sets how often data is written to tensorboard and checks train loss , by default 25.
    """
    logger = Logger(exp_ID, log_dir)
    sparsity_scheduler.path = (
        logger.log_dir
    )  # write checkpoint to same folder as tb output.

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)

    n_samples = data_train.shape[0]
    model.t.data = -torch.var(target)
    model.b.data = -torch.var(target)

    prior = torch.distributions.gamma.Gamma(n_samples / 2, n_samples / 2 * 1e-4)

    # Training
    convergence = Convergence(**convergence_kwargs)
    for iteration in torch.arange(0, max_iterations):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        # we train the log of these things since they're very big
        # we cap alpha and beta to prevent overflow
        tau_ = torch.exp(model.t).clamp(max=1e8)
        beta_ = torch.exp(model.b).clamp(max=1e8)

        MSE = torch.mean((prediction - target_train) ** 2, dim=0)  # loss per output
        Reg = torch.stack(
            [
                torch.mean((dt - theta @ coeff_vector) ** 2)
                for dt, theta, coeff_vector in zip(
                    time_derivs,
                    thetas,
                    model.constraint_coeffs(scaled=False, sparse=True),
                )
            ]
        )

        p_MSE = -n_samples / 2 * (tau_ * MSE - torch.log(tau_))
        p_reg = -n_samples / 2 * (beta_ * Reg - torch.log(beta_))
        loss = torch.sum(-p_MSE - p_reg - prior.log_prob(beta_))

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            with torch.no_grad():
                prediction_test = model.func_approx(data_test)[0]
                MSE_test = torch.mean(
                    (prediction_test - target_test) ** 2, dim=0
                )  # loss per output

            # ====================== Logging =======================
            _ = model.sparse_estimator(
                thetas, time_derivs
            )  # calculating estimator coeffs but not setting mask
            logger(
                iteration,
                loss,
                MSE,
                Reg,
                model.constraint_coeffs(sparse=True, scaled=True),
                model.constraint_coeffs(sparse=True, scaled=False),
                model.estimator_coeffs(),
                MSE_test=MSE_test,
                p_MSE=p_MSE,
                p_reg=p_reg,
                tau=tau_,
                beta_=beta_,
            )

            # ================== Sparsity update =============
            # Updating sparsity
            update_sparsity = sparsity_scheduler(
                iteration, torch.sum(MSE_test), model, optimizer
            )
            if update_sparsity:
                model.constraint.sparsity_masks = model.sparse_estimator(
                    thetas, time_derivs
                )

            # ================= Checking convergence
            l1_norm = torch.sum(
                torch.abs(
                    torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)
                )
            )
            converged = convergence(iteration, l1_norm)
            if converged:
                break
    logger.close(model)


def train_bayes_full(
    model: DeepMoD,
    data: torch.Tensor,
    target: torch.Tensor,
    optimizer,
    sparsity_scheduler,
    split=0.8,
    exp_ID: str = None,
    log_dir: str = None,
    max_iterations: int = 10000,
    write_iterations: int = 25,
    **convergence_kwargs
) -> None:
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
    sparsity_scheduler.path = (
        logger.log_dir
    )  # write checkpoint to same folder as tb output.

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)

    n_samples = data_train.shape[0]
    s0 = torch.tensor(1e-4)
    a = torch.tensor(1.0)

    # Training
    convergence = Convergence(**convergence_kwargs)
    for iteration in torch.arange(0, max_iterations):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        # Calculate MSE and reg, assuming one output
        MSE = torch.mean((prediction - target_train) ** 2, dim=0)  # loss per output
        Reg = torch.stack(
            [
                torch.mean((dt - theta @ coeff_vector) ** 2)
                for dt, theta, coeff_vector in zip(
                    time_derivs,
                    thetas,
                    model.constraint_coeffs(scaled=False, sparse=True),
                )
            ]
        )

        p_MSE = log_prob_students_t_pytorch(
            x=prediction,
            v=torch.tensor(n_samples, dtype=torch.float32),
            mu=target_train,
            sigma=torch.sum(MSE),
        )

        mu_reg = thetas[0] @ model.constraint_coeffs(scaled=False, sparse=True)[0]
        sigma_reg = (a * s0 + torch.sum(Reg)) / (a + 1)
        p_reg = log_prob_students_t_pytorch(
            x=time_derivs[0], v=(a + 1) * n_samples, mu=mu_reg, sigma=sigma_reg
        )

        loss = -p_reg - p_MSE
        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            with torch.no_grad():
                prediction_test = model.func_approx(data_test)[0]
                MSE_test = torch.mean(
                    (prediction_test - target_test) ** 2, dim=0
                )  # loss per output

            # ====================== Logging =======================
            _ = model.sparse_estimator(
                thetas, time_derivs
            )  # calculating estimator coeffs but not setting mask
            logger(
                iteration,
                loss,
                MSE,
                Reg,
                model.constraint_coeffs(sparse=True, scaled=True),
                model.constraint_coeffs(sparse=True, scaled=False),
                model.estimator_coeffs(),
                MSE_test=MSE_test,
                p_MSE=p_MSE,
                p_reg=p_reg,
            )

            # ================== Sparsity update =============
            # Updating sparsity
            update_sparsity = sparsity_scheduler(
                iteration, torch.sum(MSE_test), model, optimizer
            )
            if update_sparsity:
                model.constraint.sparsity_masks = model.sparse_estimator(
                    thetas, time_derivs
                )

            # ================= Checking convergence
            l1_norm = torch.sum(
                torch.abs(
                    torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)
                )
            )
            converged = convergence(iteration, l1_norm)
            if converged:
                break
    logger.close(model)


def train_SBL(
    model: DeepMoD,
    data: torch.Tensor,
    target: torch.Tensor,
    optimizer,
    sparsity_scheduler,
    split=0.8,
    exp_ID: str = None,
    log_dir: str = None,
    max_iterations: int = 10000,
    write_iterations: int = 25,
    **convergence_kwargs
) -> None:
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
    sparsity_scheduler.path = (
        logger.log_dir
    )  # write checkpoint to same folder as tb output.

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)

    n_samples = data_train.shape[0]
    model.t.data = -torch.log(torch.var(target))
    model.b.data = -torch.log(torch.var(target))

    prior = torch.distributions.gamma.Gamma(n_samples / 2, n_samples / 2 * 1e-4)

    # Training
    convergence = Convergence(**convergence_kwargs)
    for iteration in torch.arange(0, max_iterations):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        tau_ = torch.exp(model.t).clamp(max=1e8)
        alpha_ = torch.exp(model.a).clamp(max=1e8)
        beta_ = torch.exp(model.b).clamp(max=1e8)

        MSE = torch.mean((prediction - target_train) ** 2, dim=0)  # loss per output
        Reg = torch.stack(
            [
                torch.mean((dt - theta @ coeff_vector) ** 2)
                for dt, theta, coeff_vector in zip(
                    time_derivs,
                    thetas,
                    model.constraint_coeffs(scaled=False, sparse=True),
                )
            ]
        )

        mask = model.constraint.sparsity_masks[0]
        p_reg, mn = SBL(thetas[0], time_derivs[0], mask, alpha_, beta_)

        p_MSE = -n_samples / 2 * (tau_ * MSE - torch.log(tau_))
        loss = torch.sum(-p_MSE - p_reg - prior.log_prob(beta_))

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            with torch.no_grad():
                prediction_test = model.func_approx(data_test)[0]
                MSE_test = torch.mean(
                    (prediction_test - target_test) ** 2, dim=0
                )  # loss per output

            # ====================== Logging =======================
            _ = model.sparse_estimator(
                thetas, time_derivs
            )  # calculating estimator coeffs but not setting mask
            logger(
                iteration,
                loss,
                MSE,
                Reg,
                model.constraint_coeffs(sparse=True, scaled=True),
                model.constraint_coeffs(sparse=True, scaled=False),
                model.estimator_coeffs(),
                MSE_test=MSE_test,
                p_MSE=p_MSE,
                p_reg=p_reg,
                tau=tau_,
                alpha=alpha_,
                beta_=beta_,
                mn=mn,
            )

            # ================== Sparsity update =============
            # Updating sparsity
            update_sparsity = sparsity_scheduler(
                iteration, torch.sum(MSE_test), model, optimizer
            )
            if update_sparsity:
                model.constraint.sparsity_masks = model.sparse_estimator(
                    thetas, time_derivs
                )

            # ================= Checking convergence
            l1_norm = torch.sum(
                torch.abs(
                    torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)
                )
            )
            converged = convergence(iteration, l1_norm)
            if converged:
                break
    logger.close(model)


def SBL(X, y, mask, alpha_, beta_):
    n_samples = X.shape[0]

    # Calculating coeff vectors
    X_normed = X[:, mask] / torch.norm(X[:, mask], dim=0, keepdim=True)
    alpha = alpha_[mask]

    A_inv = torch.inverse(torch.diag(alpha) + beta_ * X_normed.T @ X_normed)
    mn = beta_ * A_inv @ X_normed.T @ y

    # Calculating likelihood
    p = (
        -beta_ * torch.sum((y - X_normed @ mn) ** 2)
        - torch.sum(alpha[:, None] * mn ** 2)
        + n_samples * torch.log(beta_)
        + torch.sum(torch.log(alpha))
        + torch.logdet(A_inv)
    )

    # Putting 0's in right spot
    coeffs = (
        torch.zeros((mask.shape[0], 1)).to(X.device).masked_scatter_(mask[:, None], mn)
    )

    return p, coeffs


def log_prob_students_t(x, v, mu, sigma):
    """ Only works for 1D"""

    n_samples = x.shape[0]  # number of samples
    z = 1 / sigma * torch.sum((x - mu) ** 2)  # normalized coordinates

    log_p = (
        -(v + 1) / 2 * torch.log(1 + 1 / v * z)
        - n_samples / 2 * torch.log(sigma)
        + torch.lgamma((v + 1) / 2)
        - torch.lgamma(v / 2)
        - 1 / 2 * torch.log(v)  # should be n_samples as well?; check!
        - 1 / 2 * np.log(np.pi)
    )

    return log_p


def log_prob_students_t_pytorch(x, v, mu, sigma):
    """ Only works for 1D"""

    z = x - mu
    dist = torch.distributions.studentT.StudentT(df=v, loc=0.0, scale=torch.sqrt(sigma))
    log_prob = torch.sum(dist.log_prob(z))

    return log_prob

