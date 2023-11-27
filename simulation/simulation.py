from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import taichi as ti

from generator import HistoricalMonthlyGenerator


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
            investment_return_gen: HistoricalMonthlyGenerator,
            safe_deposit_rate_gen: HistoricalMonthlyGenerator,
            inflation_rate_gen: HistoricalMonthlyGenerator
    ):
        self.num_sim = num_sim
        self.capital_gains_tax_rate = capital_gains_tax_rate
        self.investment_tax_exemption = investment_tax_exemption
        self.factors_gen = _create_factors_gen(
            investment_return_gen, safe_deposit_rate_gen, inflation_rate_gen
        )
        self.values: ti.Field

    def __call__(
            self, num_years: int,
            current_invest: float, current_save: float,
            monthly_invest: float, monthly_save: float
    ) -> SimulationResult:
        self._create_fields(num_years)
        self._simulate(
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
            value=values[:, :, 0],
            value_only_safe_deposit=values[:, :, 1]
        )

    def _create_fields(self, num_years: int) -> None:
        self.values = ti.field(dtype=float, shape=(self.num_sim, 12*num_years, 2))

    @ti.kernel
    def _simulate(
            self,
            current_invest: float, current_save: float,
            monthly_invest: float, monthly_save: float
    ):
        for i in range(self.num_sim):  # Parallel
            acc_invest = acc_total_invested = current_invest + monthly_invest
            acc_save = acc_total_saved = current_save + monthly_save
            acc_only_save = acc_invest + acc_save
            acc_inflation = 1.0

            for j in range(self.num_months):  # Sequential
                # Portfolio value before taxes and inflation
                value = acc_invest + acc_save
                value_only_safe = acc_only_save

                # Portfolio value after taxes and before inflation
                profit_invest = acc_invest - acc_total_invested
                profit_save = acc_save - acc_total_saved
                profit_only_save = value_only_safe - (acc_total_invested + acc_total_saved)

                # Losses between ETFs and bonds/savings accounts are settled
                tax_invest = max(0., profit_invest+min(0., profit_save)) \
                    * (1.-self.investment_tax_exemption) * self.capital_gains_tax_rate
                tax_safe = max(0., profit_save+min(0., profit_invest)) * self.capital_gains_tax_rate
                tax_only_save = max(0., profit_only_save) * self.capital_gains_tax_rate

                value -= tax_invest + tax_safe
                value_only_safe -= tax_only_save

                # Portfolio value after inflation
                value /= acc_inflation
                value_only_safe /= acc_inflation

                # Store portfolio value
                self.values[i, j, 0] = value
                self.values[i, j, 1] = value_only_safe

                # Update accumulators
                invest_increase, safe_increase, inflation_increase = self.factors_gen.sample()
                acc_invest = (acc_invest + monthly_invest) * invest_increase
                acc_save = (acc_save + monthly_save) * safe_increase
                acc_only_save = (acc_only_save + monthly_invest + monthly_save) * safe_increase
                acc_inflation *= inflation_increase

    @property
    def num_months(self) -> int:
        return self.values.shape[1]


def _create_factors_gen(
        investment_return_gen: HistoricalMonthlyGenerator,
        safe_deposit_rate_gen: HistoricalMonthlyGenerator,
        inflation_rate_gen: HistoricalMonthlyGenerator
) -> "FactorsGen":
    inv_values = investment_return_gen.values
    safe_values = safe_deposit_rate_gen.values
    inflation_values = inflation_rate_gen.values

    common_idx = inv_values.index.intersection(safe_values.index.intersection(inflation_values.index))
    return FactorsGen(
        inv_values[common_idx], safe_values[common_idx], inflation_values[common_idx]
    )


@ti.data_oriented
class FactorsGen:
    def __init__(
            self,
            hist_investment_returns: Sequence[float],
            hist_safe_deposit_rate: Sequence[float],
            hist_inflation_rates: Sequence[float],
    ):
        self.hist_investment_returns = _convert(hist_investment_returns)
        self.hist_safe_deposit_rate = _convert(hist_safe_deposit_rate)
        self.hist_inflation_rates = _convert(hist_inflation_rates)

    @ti.func
    def sample(self) -> tuple[float, float, float]:
        idx = int(ti.floor(self.hist_investment_returns.shape[0] * ti.random()))
        return self.hist_investment_returns[idx],  \
            self.hist_safe_deposit_rate[idx], \
            self.hist_inflation_rates[idx]


def _convert(seq: Sequence[float]) -> ti.Field:
    x = np.array(seq)
    f = ti.field(float, shape=x.shape)
    f.from_numpy(x)
    return f


if __name__ == '__main__':
    from generator import (
        HistoricalACWIIMIReturns,
        Historical1YearUSBondYields,
        HistoricalGermanInflation
    )

    sim = MonteCarloSimulation(
        num_sim=100_000,
        capital_gains_tax_rate=0.278186,
        investment_tax_exemption=0.3,
        investment_return_gen=HistoricalACWIIMIReturns(),
        safe_deposit_rate_gen=Historical1YearUSBondYields(),
        inflation_rate_gen=HistoricalGermanInflation()
    )

    from time import time

    for i in range(10):
        t = time()
        res = sim(
            num_years=200,
            current_invest=10_000, current_save=10_000,
            monthly_invest=1_000, monthly_save=1_000
        )
        print(f'Run {i}, duration: {time()-t}')
        del res

    pass
