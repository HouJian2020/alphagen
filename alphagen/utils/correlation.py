import torch
from torch import Tensor



def _mask_either_nan(x: Tensor, y: Tensor, fill_with: float = torch.nan):
    x = x.clone()  # [days, stocks]
    y = y.clone()  # [days, stocks]
    nan_mask = x.isnan() | y.isnan()
    x[nan_mask] = fill_with
    y[nan_mask] = fill_with
    n = (~nan_mask).sum(dim=1)
    return x, y, n, nan_mask


def _rank_data(x: Tensor, nan_mask: Tensor) -> Tensor:
    nan_mask = torch.isnan(x) | nan_mask
    n_valid = torch.sum(~nan_mask, dim=1, keepdim=True)
    x = torch.where(nan_mask, torch.inf, x)
    sorted_sequence = torch.sort(x)[0]
    s1 = torch.searchsorted(sorted_sequence, x, right=True)
    s2 = torch.searchsorted(sorted_sequence, x, right=False)
    x_rank = (s1 + s2) / 2
    x_rank[nan_mask] = torch.nan
    return x_rank / n_valid  # [d, s]


def _batch_pearsonr_given_mask(
        x: Tensor, y: Tensor,
        n: Tensor, mask: Tensor
) -> Tensor:
    x_demean = x - torch.nanmean(x, dim=1, keepdim=True)
    y_demean = y - torch.nanmean(y, dim=1, keepdim=True)
    cov = torch.nansum(x_demean * y_demean, dim=-1)
    x_var = torch.nansum(x_demean * x_demean, dim=-1)
    y_var = torch.nansum(y_demean * y_demean, dim=-1)
    std = (x_var * y_var) ** 0.5
    corrs = cov / std
    l_nan = std <= 1e-6
    corrs[l_nan] = torch.nan
    return corrs


def batch_spearmanr(x: Tensor, y: Tensor) -> Tensor:
    x, y, n, nan_mask = _mask_either_nan(x, y)
    rx = _rank_data(x, nan_mask)
    ry = _rank_data(y, nan_mask)
    return _batch_pearsonr_given_mask(rx, ry, n, nan_mask)


def batch_pearsonr(x: Tensor, y: Tensor) -> Tensor:
    return _batch_pearsonr_given_mask(*_mask_either_nan(x, y, fill_with=torch.nan))
