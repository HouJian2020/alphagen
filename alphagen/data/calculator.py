from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union
from torch import Tensor

from alphagen.data.expression import Expression


class AlphaCalculator(metaclass=ABCMeta):
    @abstractmethod
    def calc_alpha(self, expr: [Expression, Tensor]) -> Tensor:
        'calculate alpha'

    @abstractmethod
    def calc_single_IC_ret(self, expr: Union[Expression, Tensor]) -> float:
        'Calculate IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_rIC_ret(self, expr: Union[Expression, Tensor]) -> float:
        'Calculate Rank IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_all_ret(self, expr: Union[Expression, Tensor]) -> Tuple[float, float]:
        'Calculate both IC and Rank IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_mutual_IC(self, expr1: Union[Expression, Tensor], expr2: Union[Expression, Tensor]) -> float:
        'Calculate IC between two alphas.'

    @abstractmethod
    def calc_pool_IC_ret(self, exprs: List[Union[Expression, Tensor]], weights: List[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_rIC_ret(self, exprs: List[Union[Expression, Tensor]], weights: List[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate Rank IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_all_ret(self, exprs: List[Union[Expression, Tensor]], weights: List[float]) -> Tuple[float, float]:
        'First combine the alphas linearly,'
        'then Calculate both IC and Rank IC between the linear combination and a predefined target.'
