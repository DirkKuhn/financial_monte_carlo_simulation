from collections.abc import Sequence

import numpy as np
import taichi as ti

from simulation.generator import HistoricalMonthlyGenerator, IndependentMonthlyGenerator


@ti.data_oriented
class FactorsGen:
    """
    Taichi statically compiles code. Therefore, some tricks are required to achieve a similar flexibility common
    to Python. If multiple ``HistoricalMonthlyGenerators`` are passed, samples are drawn jointly from them.
    """
    def __init__(
            self,
            investment_return_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator,
            safe_deposit_rate_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator,
            inflation_rate_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator
    ):
        self.use_ind_inv_ret_gen = isinstance(investment_return_gen, IndependentMonthlyGenerator)
        self.use_ind_safe_deposit_rate_gen = isinstance(safe_deposit_rate_gen, IndependentMonthlyGenerator)
        self.use_ind_inflation_rate_gen = isinstance(inflation_rate_gen, IndependentMonthlyGenerator)

        self.all_independent = self.use_ind_inv_ret_gen \
            and self.use_ind_inflation_rate_gen \
            and self.use_ind_inflation_rate_gen

        self.ind_inv_return_gen = investment_return_gen if self.use_ind_inv_ret_gen else PlaceHolder()
        self.ind_safe_deposit_rate_gen = safe_deposit_rate_gen if self.use_ind_safe_deposit_rate_gen else PlaceHolder()
        self.ind_inflation_rate_gen = inflation_rate_gen if self.use_ind_inflation_rate_gen else PlaceHolder()

        self.length_values = 0

        self.hist_inv_returns = None if self.use_ind_inv_ret_gen else investment_return_gen.values
        self.hist_safe_deposit_rates = None if self.use_ind_safe_deposit_rate_gen else safe_deposit_rate_gen.values
        self.hist_inflation_rates = None if self.use_ind_inflation_rate_gen else inflation_rate_gen.values

        values = [v for v in [self.hist_inv_returns, self.hist_safe_deposit_rates, self.hist_inflation_rates]
                  if v is not None]
        if values:
            common_idx = values[0].index
            for i in range(1, len(values)):
                common_idx = common_idx.intersection(values[i].index)
            self.length_values = len(common_idx)
            assert self.length_values > 0, "Indices are disjunctive!"

        self.hist_inv_returns = ti.field(dtype=float, shape=()) \
            if self.use_ind_inv_ret_gen else _convert(self.hist_inv_returns[common_idx])
        self.hist_safe_deposit_rates = ti.field(dtype=float, shape=()) \
            if self.use_ind_safe_deposit_rate_gen else _convert(self.hist_safe_deposit_rates[common_idx])
        self.hist_inflation_rates = ti.field(dtype=float, shape=()) \
            if self.use_ind_inflation_rate_gen else _convert(self.hist_inflation_rates[common_idx])

    @ti.func
    def sample(self) -> tuple[float, float, float]:
        res_inv = self.ind_inv_return_gen.sample() if ti.static(self.use_ind_inv_ret_gen) else 0.
        res_safe = self.ind_safe_deposit_rate_gen.sample() if ti.static(self.use_ind_safe_deposit_rate_gen) else 0.
        res_inflation = self.ind_inflation_rate_gen.sample() if ti.static(self.use_ind_inflation_rate_gen) else 0.

        if ti.static(not self.all_independent):
            idx = int(ti.floor(self.length_values * ti.random()))
            if ti.static(not self.use_ind_inv_ret_gen):
                res_inv = self.hist_inv_returns[idx]
            if ti.static(not self.use_ind_safe_deposit_rate_gen):
                res_safe = self.hist_safe_deposit_rates[idx]
            if ti.static(not self.use_ind_inflation_rate_gen):
                res_inflation = self.hist_inflation_rates[idx]

        return res_inv, res_safe, res_inflation


class PlaceHolder(IndependentMonthlyGenerator):
    @ti.func
    def sample(self) -> float:
        return 0.


def _convert(seq: Sequence[float]) -> ti.Field:
    x = np.array(seq, dtype=np.float32)
    f = ti.field(dtype=float, shape=x.shape)
    f.from_numpy(x)
    return f
