import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from typing import List, Callable, Generator, Any, Literal


class AbstractModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self, dtype: Literal[torch.float, torch.double] = torch.float):
        super().__init__()
        self.dtype = dtype
    
    @abstractmethod
    def forward(self, option_params: Tensor, simul_results: List[Tensor]) -> Tensor:
        pass


class DeltaHedgingModel(nn.Module):
    def __init__(self, option_model: AbstractModel, *hedging_models: AbstractModel):
        super().__init__()
        self.option_model = option_model
        self.hedging_models = nn.ModuleList(hedging_models)


class HedgingResiduals():
    def __init__(self, option_params: Tensor):
        self.option_params = option_params
    
    def set_simulation(self, simul_func: Callable[[Tensor, List[Tensor]], Tensor], **simul_params: Any) -> None:
        self.simul_func, self.simul_params = simul_func, simul_params
    
    def set_dhedges(self, *dhedge_funcs: Callable[[Tensor, List[Tensor]], Tensor]) -> None:
        self.dhedge_funcs = dhedge_funcs
    
    def perform_simulation(self) -> None:
        simul_results = self.simul_func(self.option_params.cpu(), **self.simul_params)
        self.simul_results = list(map(lambda sr: sr.to(device=self.option_params.device), simul_results))
    
    def create_dhedges(self) -> None:
        self.dhedges = [h(self.option_params, self.simul_results) for h in self.dhedge_funcs]
    
    def __call__(self, model: DeltaHedgingModel) -> Tensor:
        assert len(self.dhedges) == len(model.hedging_models), "number of hedges != number of hedging models"
        c = model.option_model(self.option_params, self.simul_results)
        total_hedge = sum(m(self.option_params, self.simul_results) * dh for m, dh in zip(model.hedging_models, self.dhedges))
        return c[:, 1:] - c[:, :-1] - total_hedge


class BatchedHedgingResiduals(HedgingResiduals):
    def __init__(self, option_params: Tensor):
        self.option_params = option_params.cpu()
    
    def __call__(self, model: DeltaHedgingModel, nb: int = 1) -> Generator[Tensor, None, None]:
        assert len(self.dhedges) == len(model.hedging_models), f"number of hedges != number of hedging models"
        n = self.option_params.shape[1]
        
        for cb in range(nb):
            # get batch of data
            start, end = int(cb * n/nb), min(int((cb+1) * n/nb), n)
            option_params_cuda = self.option_params[:, start:end].cuda()
            simul_results_cuda = list(map(lambda sr: sr[start:end, :].cuda(), self.simul_results))
            dhedges_cut = list(map(lambda dhedge: dhedge[start:end, :], self.dhedges))
            
            # run data through models, take difference
            c = model.option_model(option_params_cuda, simul_results_cuda).cpu()
            total_hedge_cuda = sum(m(option_params_cuda, simul_results_cuda).to(device='cpu') * dh for m, dh in zip(model.hedging_models, dhedges_cut))
            yield c[:, 1:] - c[:, :-1] - total_hedge_cuda
            
            # delete variables to save space
            del option_params_cuda, simul_results_cuda, dhedges_cut, c, total_hedge_cuda


class GenerationHedgingResiduals(HedgingResiduals):
    def __init__(self, option_params: Tensor, rng_seed: int = 12345):
        super().__init__(option_params.cuda())
        torch.manual_seed(rng_seed)

    def perform_simulation(self) -> None:
        simul_results = self.simul_func(self.option_params.cpu(), **self.simul_params, use_gpu=True, rng_seed=torch.randint(10**9, (1,)).item())
        self.simul_results = list(map(lambda sr: sr.to(device=self.option_params.device), simul_results))