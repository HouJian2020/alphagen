from typing import Tuple, Optional
import gymnasium as gym
import math

from alphagen.config import MAX_EXPR_LENGTH
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.models.alpha_pool import AlphaPoolBase, AlphaPool
from alphagen.utils import reseed_everything


class AlphaEnvCore(gym.Env):
    pool: AlphaPoolBase
    _tokens: List[Token]
    _builder: ExpressionBuilder
    _print_expr: bool

    def __init__(self,
                 pool: AlphaPoolBase,
                 device: torch.device = torch.device('cuda:0'),
                 print_expr: bool = False
                 ):
        super().__init__()

        self.pool = pool
        self._print_expr = print_expr
        self._device = device

        self.eval_cnt = 0
        self.alpha_set: dict = {} # 因子名称：【因子ic, 已测试次数】

        self.render_mode = None

    def reset(
            self, *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None
    ) -> Tuple[List[Token], dict]:
        reseed_everything(seed)
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        return self._tokens, self._valid_action_types()

    def step(self, action: Token) -> Tuple[List[Token], float, bool, bool, dict]:
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            reward = self._evaluate()
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            reward = 0.0
        else:
            done = True
            reward = self._evaluate() if self._builder.is_valid() else -1.

        if math.isnan(reward):
            reward = 0.

        truncated = False  # Fk gymnasium
        return self._tokens, reward, done, truncated, self._valid_action_types()

    def _evaluate(self):
        expr: Expression = self._builder.get_tree()
        def show_log():
            if self._print_expr:
                print(expr, self.alpha_set.get(str(expr)))

        repeat = str(expr) in self.alpha_set
        n_repeat = self.alpha_set[str(expr)][1] if repeat else 0
        punishment = -n_repeat / 10  # 重复因子的惩罚

        if repeat and abs(self.alpha_set[str(expr)][0]) <= 0.0025:
            show_log()
            return punishment  # 单因子效果很差, 返回惩罚，不要继续测试了

        try:
            ret, ic, increase = self.pool.try_new_expr(expr)
            if not repeat:
                self.alpha_set[str(expr)] = [ic, 1]
            else:
                self.alpha_set[str(expr)][1] += 1
                self.alpha_set[str(expr)][0] = ic
            self.eval_cnt += 1
            show_log()
            return ret if increase else punishment  # 如果因子效果好则返回因子，否则返回惩罚
        except OutOfDataRangeError:
            show_log()
            return punishment

    def _valid_action_types(self) -> dict:
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_pair_rolling = self._builder.validate_op(PairRollingOperator)

        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_feature()
        valid_stop = self._builder.is_valid()

        ret = {
            'select': [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            'op': {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                PairRollingOperator: valid_op_pair_rolling
            }
        }
        return ret

    def valid_action_types(self) -> dict:
        return self._valid_action_types()

    def render(self, mode='human'):
        pass
