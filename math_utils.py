import torch
from torch import Tensor
from typing import Tuple, Optional


def black_scholes_5p(
    p: Tensor,
    k: Tensor,
    r: Tensor,
    m: Tensor,
    sigma: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Black-Scholes formula adapted for matricies p and m
    generated by geometric brownian motion.
    Expects all inputs to be same-shape tensors, and all tensors
    in output have that same shape.
    Exception: if some tensors are 1D and some are 2D, will extend
    1D tensors along second axis with same values (that is, will turn 
    x[i] into x'[i, _] = x[i]; assumes lower dimensions compatible)
    """    
    # upscale params
    params = [p, k, r, m, sigma]
    dim = max(0 if type(x) in [int, float] else x.dim() for x in params)
    if dim != 0:
        shape = [x.shape for x in params if type(x) not in [int, float]][0]
        for i in range(len(params)):
            if type(params[i]) in [int, float]:
                params[i] = params[i] * torch.ones(shape)
            elif params[i].dim() < len(shape):
                while params[i].dim() < len(shape):
                    params[i] = params[i].unsqueeze(-1)
                params[i] = params[i].expand(shape)
    else:  # dim == 0
        for i in range(len(params)):
            params[i] = Tensor([params[i]])
    p, k, r, m, sigma = params
    
    # useful values
    eps = torch.tensor(1e-15)
    ones_like_p = torch.ones(p.shape)
    sigm = sigma * torch.sqrt(m)
    pvk = k * torch.exp(-r * m)
    
    # negative strike price case
    isNeg = (k <= 0)
    infd1 = torch.tensor(float('Inf')) * isNeg
    infd1[infd1 != infd1] = 0 # d1 = infinity
    infnd1 = 1 * isNeg # N(d1) = 1
    infres = (p - pvk) * isNeg
    pvk = abs(pvk) + 0.001 * (k == 0) # remove nonpositive entries; we will overwrite them later
    
    # normal black-scholes computation
    d1 = torch.log(p / pvk) / torch.maximum(sigm, eps) + 0.50 * sigm
    d2 = d1 - sigm
    normcdf = torch.distributions.normal.Normal(0, 1).cdf
    nd1 = normcdf(d1)
    nd2 = normcdf(d2)
    res = p * nd1 - pvk * nd2
    
    # combine cases
    return res * ~isNeg + infres, nd1 * ~isNeg + infnd1, d1 * ~isNeg + infd1


def gbm(
    p0: Tensor,
    mu: Tensor,
    sigma: Tensor,
    m0: Tensor,
    nt: int,
    rng_seed: int = 12345,
    use_gpu: bool = False,
    db: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Function to simulate geometric brownian motion.
    Returns tuple of price, db, and maturity matricies
    """
    assert p0.device == mu.device == sigma.device == m0.device
    assert p0.shape == mu.shape == sigma.shape == m0.shape
    n = len(p0)
    torch.manual_seed(rng_seed)
    
    orig_t = {'device':p0.device, 'dtype':p0.dtype}
    cast_t = {'device':'cuda' if use_gpu else p0.device, 'dtype':p0.dtype}
    if use_gpu:
        p0, mu, sigma, m0 = p0.cuda(), mu.cuda(), sigma.cuda(), m0.cuda()
    
    dt = m0 / nt
    m = torch.stack([m0x - torch.linspace(0.0, m0x, nt+1).to(**cast_t) for m0x in m0])
    if db is None:
        db = torch.diag(torch.sqrt(dt)) @ torch.randn((n, nt), device=('cuda' if use_gpu else dt.device)).to(**cast_t)
    brownian_motion = torch.diag(sigma) @ db
    drift = torch.diag( (mu - 0.5 * sigma**2) * dt ) @ torch.ones(db.shape).to(**cast_t)
    res = torch.hstack([p0.unsqueeze(-1), torch.diag(p0) @ torch.exp(brownian_motion + drift).cumprod(dim=1)])
    return res.to(**orig_t), db.to(**orig_t), m.to(**orig_t)


def black_scholes_2p(
    moneyness: Tensor, # p / pvk
    vol_to_mat: Tensor # sigma * sqrt(m)
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Standard Black-Scholes formula for vector arguments.
    Returns tuple of option price, N(d1), and d1.
    """
    # upscale params
    params = [moneyness, vol_to_mat]
    dim = max(0 if type(x) in [int, float] else x.dim() for x in params)
    if dim != 0:
        shape = [x.shape for x in params if type(x) not in [int, float]][0]
        for i in range(len(params)):
            if type(params[i]) in [int, float]:
                params[i] = params[i] * torch.ones(shape)
            elif params[i].dim() < len(shape):
                while params[i].dim() < len(shape):
                    params[i] = params[i].unsqueeze(-1)
                params[i] = params[i].expand(shape)
    moneyness, vol_to_mat = params
    
    # negative moneyness case
    isNeg = (moneyness <= 0)
    infd1 = torch.tensor(float('Inf')) * isNeg
    infd1[infd1 != infd1] = 0 # d1 = infinity
    infnd1 = 1 * isNeg # N(d1) = 1
    infres = (1 - 1/moneyness) * isNeg # output = 1 - 1 / moneyness
    moneyness = abs(moneyness) + 0.001 * (moneyness == 0) # remove nonpositive entries; we will overwrite them later
    
    # normal black-scholes computation
    d1 = torch.clamp(torch.log(moneyness), min=-1e15) / torch.clamp(vol_to_mat, min=1e-15) + 0.5 * vol_to_mat
    d2 = d1 - vol_to_mat
    normcdf = torch.distributions.normal.Normal(0, 1).cdf
    nd1, nd2 = normcdf(d1), normcdf(d2)
    res = nd1 - nd2 / moneyness
    
    # combine cases
    return res * ~isNeg + infres, nd1 * ~isNeg + infnd1, d1 * ~isNeg + infd1


# def Linear0(*size):
#     x = nn.Linear(*size)
#     nn.init.zeros_(x.weight)
#     nn.init.zeros_(x.bias)
#     return x
