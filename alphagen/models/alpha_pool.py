from itertools import count
import math
from typing import List, Optional, Tuple, Set
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import Tensor
from alphagen.data.calculator import AlphaCalculator

from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import masked_mean_std
from alphagen_qlib.stock_data import StockData


class AlphaPoolBase(metaclass=ABCMeta):
    def __init__(
            self,
            capacity: int,
            calculator: AlphaCalculator,
            device: torch.device = torch.device('cpu')
    ):
        self.capacity = capacity
        self.calculator = calculator
        self.device = device

    @abstractmethod
    def to_dict(self) -> dict: ...

    @abstractmethod
    def try_new_expr(self, expr: Expression) -> Tuple[float, float, bool]: ...

    @abstractmethod
    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]: ...


class AlphaPool(AlphaPoolBase):
    def __init__(
            self,
            capacity: int,
            calculator: AlphaCalculator,
            ic_lower_bound: Optional[float] = None,
            l1_alpha: float = 5e-3,
            device: torch.device = torch.device('cpu')
    ):
        super().__init__(capacity, calculator, device)

        self.size: int = 0
        self.exprs: List[Optional[Expression]] = [None for _ in range(capacity + 1)]  # 表达式的集合
        self.alphas: List[Optional[Tensor]] = [None for _ in range(capacity + 1)]  # 表达式对应的tensor集合
        self.single_ics: np.ndarray = np.zeros(capacity + 1)
        self.mutual_ics: np.ndarray = np.identity(capacity + 1)
        self.weights: np.ndarray = np.zeros(capacity + 1)
        self.best_ic_ret: float = -1.

        self.ic_lower_bound = ic_lower_bound or -1.
        self.l1_alpha = l1_alpha

        self.eval_cnt = 0

    @property
    def state(self) -> dict:
        return {
            "exprs": list(self.exprs[:self.size]),
            "ics_ret": list(self.single_ics[:self.size]),
            "weights": list(self.weights[:self.size]),
            "best_ic_ret": self.best_ic_ret
        }

    def to_dict(self) -> dict:
        return {
            "exprs": [str(expr) for expr in self.exprs[:self.size]],
            "weights": list(self.weights[:self.size])
        }

    def try_new_expr(self, expr: Expression) -> Tuple[float, float, bool]:
        ic_ret, ic_mut, factor = self._calc_ics(expr, ic_mut_threshold=0.99)
        if ic_ret is None or ic_mut is None or np.isnan(ic_ret) or np.isnan(ic_mut).any():
            return 0., ic_ret,  False

        self._add_factor(expr, factor, ic_ret, ic_mut)
        if self.size > 1:
            new_weights = self._optimize(alpha=self.l1_alpha, lr=5e-4, n_iter=500)
            worst_idx = np.argmin(np.abs(new_weights))
            if worst_idx != self.capacity:
                self.weights[:self.size] = new_weights
            self._pop()

        new_ic_ret = self.evaluate_ensemble()
        increment = new_ic_ret - self.best_ic_ret
        if increment > 0:
            self.best_ic_ret = new_ic_ret
        self.eval_cnt += 1
        return new_ic_ret, ic_ret,  increment > 1e-7

    def force_load_exprs(self, exprs: List[Expression]) -> None:
        for expr in exprs:
            ic_ret, ic_mut, factor = self._calc_ics(expr, ic_mut_threshold=None)
            assert ic_ret is not None and ic_mut is not None
            self._add_factor(expr, factor, ic_ret, ic_mut)
            assert self.size <= self.capacity
        self._optimize(alpha=self.l1_alpha, lr=5e-4, n_iter=500)

    def _optimize(self, alpha: float, lr: float, n_iter: int) -> np.ndarray:
        if math.isclose(alpha, 0.):  # no L1 regularization
            return self._optimize_lstsq()  # very fast

        ics_ret = torch.from_numpy(self.single_ics[:self.size]).to(self.device)
        ics_mut = torch.from_numpy(self.mutual_ics[:self.size, :self.size]).to(self.device)
        weights = torch.from_numpy(self.weights[:self.size]).to(self.device).requires_grad_()
        optim = torch.optim.Adam([weights], lr=lr)

        loss_ic_min = 1e9 + 7  # An arbitrary big value
        best_weights = weights.cpu().detach().numpy()
        iter_cnt = 0
        for it in count():
            ret_ic_sum = (weights * ics_ret).sum()
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_ic_curr = loss_ic.item()

            loss_l1 = torch.norm(weights, p=1)  # type: ignore
            loss = loss_ic + alpha * loss_l1

            optim.zero_grad()
            loss.backward()
            optim.step()

            if loss_ic_min - loss_ic_curr > 1e-6:
                iter_cnt = 0
            else:
                iter_cnt += 1

            if loss_ic_curr < loss_ic_min:
                best_weights = weights.cpu().detach().numpy()
                loss_ic_min = loss_ic_curr

            if iter_cnt >= n_iter or it >= 10000:
                break

        return best_weights

    def _optimize_lstsq(self) -> np.ndarray:
        try:
            return np.linalg.lstsq(self.mutual_ics[:self.size, :self.size], self.single_ics[:self.size])[0]
        except (np.linalg.LinAlgError, ValueError):
            return self.weights[:self.size]

    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]:
        ic, rank_ic = calculator.calc_pool_all_ret(self.exprs[:self.size], self.weights[:self.size])
        return ic, rank_ic

    def evaluate_ensemble(self) -> float:
        ic = self.calculator.calc_pool_IC_ret(self.alphas[:self.size], self.weights[:self.size])
        return ic

    @property
    def _under_thres_alpha(self) -> bool:
        if self.ic_lower_bound is None or self.size > 1:
            return False
        return self.size == 0 or abs(self.single_ics[0]) < self.ic_lower_bound

    def _calc_ics(
            self,
            expr: Expression,
            ic_mut_threshold: Optional[float] = None
    ) -> Tuple[float, Optional[List[float]], Tensor]:
        factor = self.calculator.calc_alpha(expr)
        single_ic = self.calculator.calc_single_IC_ret(factor)
        if not self._under_thres_alpha and single_ic < self.ic_lower_bound:
            return single_ic, None, factor

        mutual_ics = []
        for i in range(self.size):
            mutual_ic = self.calculator.calc_mutual_IC(factor, self.alphas[i])
            if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                return single_ic, None, factor
            mutual_ics.append(mutual_ic)

        return single_ic, mutual_ics, factor

    def _add_factor(
            self,
            expr: Expression,
            factor: Tensor,
            ic_ret: float,
            ic_mut: List[float]
    ):
        if self._under_thres_alpha and self.size == 1:
            self._pop()
        n = self.size
        self.exprs[n] = expr
        self.alphas[n] = factor
        self.single_ics[n] = ic_ret
        for i in range(n):
            self.mutual_ics[i][n] = self.mutual_ics[n][i] = ic_mut[i]
        self.weights[n] = ic_ret  # An arbitrary init value
        self.size += 1

    def _pop(self) -> None:
        if self.size <= self.capacity:
            return
        idx = np.argmin(np.abs(self.weights))
        self._swap_idx(idx, self.capacity)
        self.size = self.capacity

    def _swap_idx(self, i, j) -> None:
        if i == j:
            return
        self.exprs[i], self.exprs[j] = self.exprs[j], self.exprs[i]
        self.alphas[i], self.alphas[j] = self.alphas[j], self.alphas[i]
        self.single_ics[i], self.single_ics[j] = self.single_ics[j], self.single_ics[i]
        self.mutual_ics[:, [i, j]] = self.mutual_ics[:, [j, i]]
        self.mutual_ics[[i, j], :] = self.mutual_ics[[j, i], :]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]
