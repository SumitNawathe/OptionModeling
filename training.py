import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Callable, Generator, Union, Any, Literal
from architecture import DeltaHedgingModel, HedgingResiduals, BatchedHedgingResiduals, GenerationHedgingResiduals


def lbfgs_training_loop(
    model: DeltaHedgingModel,
    option_params: Tensor,
    *dhedge_funcs: Callable[[Tensor, List[Tensor]], Tensor],
    nt: int = 10**3,
    ni: int = 5,
    print_loss_inner: bool = False,
    print_loss_outer: bool = True,
    sum_path: bool = True
) -> None:
    # get option param dimensions
    p, n = option_params.shape
    
    # perform simulation and hedge generation
    print(f"Simulation")
    time0 = timeit.default_timer()
    residuals = HedgingResiduals(option_params)
    residuals.set_simulation(gbm_simulation, nt=nt)
    residuals.set_dhedges(*dhedge_funcs)
    residuals.perform_simulation()
    residuals.create_dhedges()
    print(f"Time: {timeit.default_timer() - time0}")
    
    # create optimizer and loss function
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=25, max_eval=None, 
                                  tolerance_grad=1e-07, tolerance_change=1e-09, 
                                  history_size=100, line_search_fn='strong_wolfe')
    floss = torch.nn.MSELoss(reduction='sum')
    res_zero = torch.zeros((n,) if sum_path else (n, nt)).to(device=option_params.device, dtype=option_params.dtype)


    # define closure
    def closure():
        optimizer.zero_grad()
        res = residuals(model)
        if sum_path:
            res = res.sum(axis=1)
        loss = floss(res, res_zero)
        loss.backward()
        if print_loss_inner:
            print("inner loss = ", loss.item())
        return loss
    
    # perform training
    print(f"LBFGS Training")
    time0 = timeit.default_timer()
    for epoch in range(1, ni+1):
        print(f"{epoch=}")
        optimizer.step(closure)
        
        if print_loss_outer:
            with torch.no_grad():
                res = residuals(model)
                if sum_path:
                    res = res.sum(axis=1)
                loss = floss(res, res_zero)
                print("loss =", loss.item())
            
    print(f"Time: {timeit.default_timer() - time0}")


def adam_training_loop(
    model: DeltaHedgingModel,
    option_params: Tensor,
    *dhedge_funcs: Callable[[Tensor, List[Tensor]], Tensor],
    nt: int = 10**3,
    ni: int = 500,
    lr: float = 1e-2,
    print_loss_freq: int = 50,
    sum_path: bool = True
) -> None:
    # get option param dimensions
    p, n = option_params.shape
    
    # perform simulation and hedge generation
    print(f"Simulation")
    time0 = timeit.default_timer()
    residuals = HedgingResiduals(option_params)
    residuals.set_simulation(gbm_simulation, nt=nt)
    residuals.set_dhedges(*dhedge_funcs)
    residuals.perform_simulation()
    residuals.create_dhedges()
    print(f"Time: {timeit.default_timer() - time0}")
    
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    floss = torch.nn.MSELoss(reduction='sum')
    res_zero = torch.zeros((n,) if sum_path else (n, nt)).to(device=option_params.device, dtype=option_params.dtype)

    # perform training
    print("Adam optimizer:")
    time0 = timeit.default_timer()
    for epoch in range(1, ni+1):
        res = residuals(model)
        if sum_path:
            res = res.sum(axis=1)
        loss = floss(res, res_zero)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % print_loss_freq == 0:
            print(f"{epoch=}, loss={loss.item()}")
    print(f"Time: {timeit.default_timer() - time0}")


####################
## BATCHED SYSTEM ##
####################


def batched_adam_training_loop(
    model: DeltaHedgingModel,
    option_params: Tensor,
    *dhedge_funcs: Callable[[Tensor, List[Tensor]], Tensor],
    nt: int = 10**3,
    ni: int = 500,
    nb: int = 1,
    lr: float = 1e-2,
    print_loss_freq: int = 50,
    sum_path: bool = True
) -> None:
    # get option param dimensions
    p, n = option_params.shape
    
    # perform simulation and hedge generation
    print(f"Simulation")
    time0 = timeit.default_timer()
    residuals = BatchedHedgingResiduals(option_params)
    residuals.set_simulation(gbm_simulation, nt=nt)
    residuals.set_dhedges(*dhedge_funcs)
    residuals.perform_simulation()
    residuals.create_dhedges()
    print(f"Time: {timeit.default_timer() - time0}")
    
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    floss = torch.nn.MSELoss(reduction='sum')

    # perform training
    print("Adam optimizer:")
    time0 = timeit.default_timer()
    for epoch in range(1, ni+1):
        total_loss = 0.0
        optimizer.zero_grad()
        
        for cb, res in enumerate(residuals(model, nb=nb)):
            n_in_batch = min(int((cb+1) * n/nb), n) - int(cb * n/nb)
            res_zero = torch.zeros((n_in_batch,) if sum_path else (n_in_batch, nt))
            if sum_path:
                res = res.sum(axis=1)
            loss = floss(res, res_zero)
            loss.backward()
            total_loss += loss.item()
        optimizer.step()
        
        if epoch % print_loss_freq == 0:
            print(f"{epoch=}, loss={total_loss}")
    print(f"Time: {timeit.default_timer() - time0}")


def batched_lbfgs_training_loop(
    model: DeltaHedgingModel,
    option_params: Tensor,
    *dhedge_funcs: Callable[[Tensor, List[Tensor]], Tensor],
    nt: int = 10**3,
    ni: int = 5,
    nb: int = 1,
    print_loss_inner: bool = False,
    print_loss_outer: bool = True,
    sum_path: bool = True
) -> None:
    # get option param dimensions
    p, n = option_params.shape
    
    # perform simulation and hedge generation
    print(f"Simulation")
    time0 = timeit.default_timer()
    residuals = BatchedHedgingResiduals(option_params)
    residuals.set_simulation(gbm_simulation, nt=nt)
    residuals.set_dhedges(*dhedge_funcs)
    residuals.perform_simulation()
    residuals.create_dhedges()
    print(f"Time: {timeit.default_timer() - time0}")
    
    # create optimizer and loss function
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=25, max_eval=None, 
                                  tolerance_grad=1e-07, tolerance_change=1e-09, 
                                  history_size=100, line_search_fn='strong_wolfe')
    floss = torch.nn.MSELoss(reduction='sum')
    res_zero = torch.zeros((n, nt)).to(device=option_params.device, dtype=option_params.dtype)

    # define closure
    def closure():
        optimizer.zero_grad()
        total_loss = 0.0
        
        for cb, res in enumerate(residuals(model, nb=nb)):
            n_in_batch = min(int((cb+1) * n/nb), n) - int(cb * n/nb)
            res_zero = torch.zeros((n_in_batch,) if sum_path else (n_in_batch, nt))
            if sum_path:
                res = res.sum(axis=1)
            loss = floss(res, res_zero)
            loss.backward()
            total_loss += loss.item()
        
        if print_loss_inner:
            print(f"inner loss={total_loss}")
        return total_loss
    
    # perform training
    print(f"LBFGS Training")
    time0 = timeit.default_timer()
    for epoch in range(1, ni):
        print(f"{epoch=}")
        optimizer.step(closure)
        
        if print_loss_outer:
            with torch.no_grad():
                total_loss = 0.0
                for cb, res in enumerate(residuals(model, nb=nb)):
                    n_in_batch = min(int((cb+1) * n/nb), n) - int(cb * n/nb)
                    res_zero = torch.zeros((n_in_batch,) if sum_path else (n_in_batch, nt))
                    if sum_path:
                        res = res.sum(axis=1)
                    total_loss += floss(res, res_zero).item()
                print("loss =", total_loss)
    print(f"Time: {timeit.default_timer() - time0}")



#######################
## GENERATION SYSTEM ##
#######################


def generation_lbfgs_training_loop(
    model: DeltaHedgingModel,
    option_params: Tensor,
    *dhedge_funcs: Callable[[Tensor, List[Tensor]], Tensor],
    nt: int = 10**3,
    ni: int = 5,
    print_loss_inner: bool = False,
    print_loss_outer: bool = True,
    sum_path: bool = True
) -> None:
    # get option param dimensions
    p, n = option_params.shape
    
    # perform simulation and hedge generation
    print(f"Simulation")
    time0 = timeit.default_timer()
    residuals = GenerationHedgingResiduals(option_params)
    residuals.set_simulation(gbm_simulation, nt=nt)
    residuals.set_dhedges(*dhedge_funcs)
    print(f"Time: {timeit.default_timer() - time0}")
    
    # create optimizer and loss function
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=25, max_eval=None, 
                                  tolerance_grad=1e-07, tolerance_change=1e-09, 
                                  history_size=100, line_search_fn='strong_wolfe')
    floss = torch.nn.MSELoss(reduction='sum')
    res_zero = torch.zeros((n,) if sum_path else (n, nt)).to(device=option_params.device, dtype=option_params.dtype)

    # define closure
    def closure():
        optimizer.zero_grad()
        res = residuals(model)
        if sum_path:
            res = res.sum(axis=1)
        loss = floss(res, res_zero)
        loss.backward()
        if print_loss_inner:
            print("inner loss = ", loss.item())
        return loss
    
    # perform training
    print(f"LBFGS Training")
    time0 = timeit.default_timer()
    for epoch in range(1, ni):
        print(f"{epoch=}")
        residuals.perform_simulation()
        residuals.create_dhedges()
        optimizer.step(closure)
        
        if print_loss_outer:
            with torch.no_grad():
                res = residuals(model)
                if sum_path:
                    res = res.sum(axis=1)
                loss = floss(res, res_zero)
                print("loss =", loss.item())
            
    print(f"Time: {timeit.default_timer() - time0}")


def generation_adam_training_loop(
    model: DeltaHedgingModel,
    option_params: Tensor,
    *dhedge_funcs: Callable[[Tensor, List[Tensor]], Tensor],
    nt: int = 10**3,
    ni: int = 500,
    lr: float = 1e-2,
    print_loss_freq: int = 50,
    sum_path: bool = True
) -> None:
    # get option param dimensions
    p, n = option_params.shape
    
    # perform simulation and hedge generation
    print(f"Simulation")
    time0 = timeit.default_timer()
    residuals = GenerationHedgingResiduals(option_params)
    residuals.set_simulation(gbm_simulation, nt=nt)
    residuals.set_dhedges(*dhedge_funcs)
    print(f"Time: {timeit.default_timer() - time0}")
    
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    floss = torch.nn.MSELoss(reduction='sum')
    res_zero = torch.zeros((n,) if sum_path else (n, nt)).to(device=option_params.device, dtype=option_params.dtype)

    # perform training
    print("Adam optimizer:")
    time0 = timeit.default_timer()
    for epoch in range(1, ni+1):
        residuals.perform_simulation()
        residuals.create_dhedges()
        res = residuals(model)
        if sum_path:
            res = res.sum(axis=1)
        loss = floss(res, res_zero)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % print_loss_freq == 0:
            print(f"{epoch=}, loss={loss.item()}")
    print(f"Time: {timeit.default_timer() - time0}")
    return loss.item()

