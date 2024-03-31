from typing import List, Union, Optional, Tuple
from enum import IntEnum
import numpy as np
import pandas as pd
import torch

from findata import get_data
class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5

IND_MAP = {'csi300': '000300',
           'csi500': '000905',
           'csi800': '000906',
           'csi1000': '000852',}


class StockData:


    def __init__(self,
                 instrument: str,
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda:0')) -> None:


        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.data, self._dates, self._stock_ids = self._get_data()


    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        trade_date: pd.Series = get_data.ATradeDate()
        start_index = np.where(trade_date >= self._start_time)[0][0]  # type: ignore
        end_index = np.where(trade_date <= self._end_time)[0][-1]  # type: ignore
        real_start_time = trade_date.iloc[start_index - self.max_backtrack_days]
        real_end_time = trade_date.iloc[end_index + self.max_future_days]

        df_weight = get_data.index_weight().query(f"indCode == '{IND_MAP[self._instrument]}'")
        df_weight.sort_values('date', inplace=True)
        df_weight['month_day'] = df_weight['date'] + pd.offsets.MonthEnd(0)
        df_weight.drop_duplicates(subset=['code', 'month_day'], keep='last', inplace=True)

        df_data = get_data.AStockPV_daily(real_start_time)
        df_data['month_day'] = df_data['tradeDate'] - pd.offsets.MonthEnd(1)
        df_data = pd.merge(df_data, df_weight[['code', 'month_day', 'weight']], on=['code', 'month_day'], how='left')

        l_tend = df_data['tradeDate'] <= real_end_time
        df_temp = df_data.loc[l_tend]
        l_code_select = df_temp.groupby('code')['weight'].transform(lambda x: (x >= 1e-6).any())
        df_temp = df_temp.loc[l_code_select]

        df_feature = df_temp[['code', 'tradeDate']].copy()
        for col in exprs:
            if col[1:] in ['open', 'high', 'low', 'close']:
                df_feature[col] = df_temp[col[1:]] * df_temp['adjFactor']
            if col == '$volume':
                df_feature[col] = df_temp['volume']
            if col == '$vwap':
                vwap = df_temp['amount'] * df_temp['adjFactor'] / df_temp['volume']
                df_feature[col] = np.where(df_temp['volume'] >= 300, vwap, df_temp['closeAdj'])

        l_null = ~(df_temp['weight'] >= 1e-6)
        df_feature.loc[l_null, exprs] = np.nan
        df_feature.set_index(['tradeDate', 'code'], inplace=True)
        return df_feature

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        df = self._load_exprs(features)
        df = df.stack().unstack(level=1)
        dates = df.index.levels[0]                                      # type: ignore
        stock_ids = df.columns
        values = df.values
        values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore
        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)


# class StockData:
#     _qlib_initialized: bool = False
#
#     def __init__(self,
#                  instrument: Union[str, List[str]],
#                  start_time: str,
#                  end_time: str,
#                  max_backtrack_days: int = 100,
#                  max_future_days: int = 30,
#                  features: Optional[List[FeatureType]] = None,
#                  device: torch.device = torch.device('cuda:0')) -> None:
#         self._init_qlib()
#
#         self._instrument = instrument
#         self.max_backtrack_days = max_backtrack_days
#         self.max_future_days = max_future_days
#         self._start_time = start_time
#         self._end_time = end_time
#         self._features = features if features is not None else list(FeatureType)
#         self.device = device
#         self.data, self._dates, self._stock_ids = self._get_data()
#
#     @classmethod
#     def _init_qlib(cls) -> None:
#         if cls._qlib_initialized:
#             return
#         import qlib
#         from qlib.config import REG_CN
#         qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_rolling", region=REG_CN)
#         cls._qlib_initialized = True
#
#     def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
#         # This evaluates an expression on the data and returns the dataframe
#         # It might throw on illegal expressions like "Ref(constant, dtime)"
#         from qlib.data.dataset.loader import QlibDataLoader
#         from qlib.data import D
#         if not isinstance(exprs, list):
#             exprs = [exprs]
#         cal: np.ndarray = D.calendar()
#         start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
#         end_index = cal.searchsorted(pd.Timestamp(self._end_time))  # type: ignore
#         real_start_time = cal[start_index - self.max_backtrack_days]
#         if cal[end_index] != pd.Timestamp(self._end_time):
#             end_index -= 1
#         real_end_time = cal[end_index + self.max_future_days]
#         return (QlibDataLoader(config=exprs)  # type: ignore
#                 .load(self._instrument, real_start_time, real_end_time))
#
#     def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
#         features = ['$' + f.name.lower() for f in self._features]
#         df = self._load_exprs(features)
#         df = df.stack().unstack(level=1)
#         dates = df.index.levels[0]                                      # type: ignore
#         stock_ids = df.columns
#         values = df.values
#         values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore
#         return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids
#
#     @property
#     def n_features(self) -> int:
#         return len(self._features)
#
#     @property
#     def n_stocks(self) -> int:
#         return self.data.shape[-1]
#
#     @property
#     def n_days(self) -> int:
#         return self.data.shape[0] - self.max_backtrack_days - self.max_future_days
#
#     def make_dataframe(
#         self,
#         data: Union[torch.Tensor, List[torch.Tensor]],
#         columns: Optional[List[str]] = None
#     ) -> pd.DataFrame:
#         """
#             Parameters:
#             - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
#             a list of tensors of size `(n_days, n_stocks)`
#             - `columns`: an optional list of column names
#             """
#         if isinstance(data, list):
#             data = torch.stack(data, dim=2)
#         if len(data.shape) == 2:
#             data = data.unsqueeze(2)
#         if columns is None:
#             columns = [str(i) for i in range(data.shape[2])]
#         n_days, n_stocks, n_columns = data.shape
#         if self.n_days != n_days:
#             raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
#                              f"match that of the current StockData ({self.n_days})")
#         if self.n_stocks != n_stocks:
#             raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
#                              f"match that of the current StockData ({self.n_stocks})")
#         if len(columns) != n_columns:
#             raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
#                              f"tensor feature count ({data.shape[2]})")
#         if self.max_future_days == 0:
#             date_index = self._dates[self.max_backtrack_days:]
#         else:
#             date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
#         index = pd.MultiIndex.from_product([date_index, self._stock_ids])
#         data = data.reshape(-1, n_columns)
#         return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
