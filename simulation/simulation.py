from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import taichi as ti

from generator import HistoricalMonthlyGenerator, IndependentMonthlyGenerator


class SimulationResult(NamedTuple):
    num_years: int
    current_invest: float
    current_save: float
    monthly_invest: float
    monthly_save: float
    value: np.ndarray
    value_only_safe_deposit: np.ndarray


ti.init()


@ti.data_oriented
class MonteCarloSimulation:
    def __init__(
            self,
            num_sim: int,
            capital_gains_tax_rate: float,
            investment_tax_exemption: float,
            investment_return_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator,
            safe_deposit_rate_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator,
            inflation_rate_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator
    ):
        self.num_sim = num_sim
        self.capital_gains_tax_rate = capital_gains_tax_rate
        self.investment_tax_exemption = investment_tax_exemption
        self.factors_gen = FactorsGen(
            investment_return_gen, safe_deposit_rate_gen, inflation_rate_gen
        )
        self.values = ti.field(dtype=float, shape=(self.num_sim, 2))

    def __call__(
            self, num_years: int,
            current_invest: float, current_save: float,
            monthly_invest: float, monthly_save: float
    ) -> SimulationResult:
        self._simulate(
            num_months=num_years*12,
            current_invest=current_invest, current_save=current_save,
            monthly_invest=monthly_invest, monthly_save=monthly_save
        )
        values = self.values.to_numpy()
        return SimulationResult(
            num_years=num_years,
            current_invest=current_invest,
            current_save=current_save,
            monthly_invest=monthly_invest,
            monthly_save=monthly_save,
            value=values[:, 0],
            value_only_safe_deposit=values[:, 1]
        )

    @ti.kernel
    def _simulate(
            self, num_months: int,
            current_invest: float, current_save: float,
            monthly_invest: float, monthly_save: float
    ):
        for i in range(self.num_sim):  # Parallel
            acc_invest = current_invest
            acc_save = current_save
            acc_only_save = acc_invest + acc_save
            acc_inflation = 1.0

            for j in range(num_months):  # Sequential
                # Update accumulators
                invest_increase, safe_increase, inflation_increase = self.factors_gen.sample()
                acc_invest = (acc_invest + monthly_invest) * invest_increase
                acc_save = (acc_save + monthly_save) * safe_increase
                acc_only_save = (acc_only_save + monthly_invest + monthly_save) * safe_increase
                acc_inflation *= inflation_increase

            # Portfolio value before taxes and inflation
            value = acc_invest + acc_save
            value_only_safe = acc_only_save

            # Amount paid
            invest_payment = current_invest + num_months * monthly_invest
            save_payment = current_save + num_months * monthly_save

            # Portfolio value after taxes and before inflation
            profit_invest = acc_invest - invest_payment
            profit_save = acc_save - save_payment
            profit_only_save = value_only_safe - (invest_payment + save_payment)

            # Losses between ETFs and bonds/savings accounts are settled
            tax_invest = max(0., profit_invest + min(0., profit_save)) \
                * (1. - self.investment_tax_exemption) * self.capital_gains_tax_rate
            tax_safe = max(0., profit_save + min(0., profit_invest)) * self.capital_gains_tax_rate
            tax_only_save = max(0., profit_only_save) * self.capital_gains_tax_rate

            value -= tax_invest + tax_safe
            value_only_safe -= tax_only_save

            # Portfolio value after inflation
            value /= acc_inflation
            value_only_safe /= acc_inflation

            # Store portfolio value
            self.values[i, 0] = value
            self.values[i, 1] = value_only_safe


class PlaceHolder(IndependentMonthlyGenerator):
    @ti.func
    def sample(self) -> float:
        return 0.


@ti.data_oriented
class FactorsGen:
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


def _convert(seq: Sequence[float]) -> ti.Field:
    x = np.array(seq)
    f = ti.field(dtype=float, shape=x.shape)
    f.from_numpy(x)
    return f


if __name__ == '__main__':
    from generator import (
        HistoricalACWIIMIReturns,
        Historical1YearUSBondYields,
        HistoricalGermanInflation
    )

    sim = MonteCarloSimulation(
        num_sim=300_000,
        capital_gains_tax_rate=0.278186,
        investment_tax_exemption=0.3,
        investment_return_gen=HistoricalACWIIMIReturns(),
        safe_deposit_rate_gen=Historical1YearUSBondYields(),
        inflation_rate_gen=HistoricalGermanInflation()
    )

    from time import time

    for i in range(2):
        t = time()
        res = sim(
            num_years=100,
            current_invest=10_000, current_save=10_000,
            monthly_invest=1_000, monthly_save=1_000
        )
        print(f'Run {i}, duration: {time()-t}')
