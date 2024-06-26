from typing import List, Optional, Tuple, Union
from torch import Tensor
import torch
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.stock_data import StockData


class QLibStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression], mask: Optional[Expression]):
        self.data = data
        self.mask = mask.evaluate(self.data)
        if target is None:  # Combination-only mode
            self.target_value = None
        else:
            self.target_value = target.evaluate(self.data)
            self.target_value[self.mask] = torch.nan
            self.target_value = normalize_by_day(self.target_value)

    def calc_alpha(self, expr: Union[Expression, Tensor]) -> Tensor:
        if isinstance(expr, Tensor):
            # 如果是tensor则为因子值而不是表达式
            return expr.clone()
        alpha = expr.evaluate(self.data)
        alpha[self.mask] = torch.nan
        return normalize_by_day(alpha)

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return torch.nanmean(batch_pearsonr(value1, value2)).item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return torch.nanmean(batch_spearmanr(value1, value2)).item()

    def make_ensemble_alpha(self, exprs: List[Union[Expression, Tensor]], weights: List[float]) -> Tensor:
        n = len(exprs)
        factors: List[Tensor] = [self.calc_alpha(exprs[i]) * weights[i] for i in range(n)]
        return sum(factors)  # type: ignore

    def calc_single_IC_ret(self, expr: Union[Expression, Tensor]) -> float:
        value = self.calc_alpha(expr)
        return self._calc_IC(value, self.target_value)

    def calc_single_rIC_ret(self, expr: Union[Expression, Tensor]) -> float:
        value = self.calc_alpha(expr)
        return self._calc_rIC(value, self.target_value)

    def calc_single_all_ret(self, expr: Union[Expression, Tensor]) -> Tuple[float, float]:
        value = self.calc_alpha(expr)
        return self._calc_IC(value, self.target_value), self._calc_rIC(value, self.target_value)

    def calc_mutual_IC(self, expr1: Union[Expression, Tensor], expr2: Union[Expression, Tensor]) -> float:
        value1, value2 = self.calc_alpha(expr1), self.calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Union[Expression, Tensor]], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(ensemble_value, self.target_value)

    def calc_pool_rIC_ret(self, exprs: List[Union[Expression, Tensor]], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_rIC(ensemble_value, self.target_value)

    def calc_pool_all_ret(self, exprs: List[Union[Expression, Tensor]], weights: List[float]) -> Tuple[float, float]:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(ensemble_value, self.target_value), self._calc_rIC(ensemble_value, self.target_value)
