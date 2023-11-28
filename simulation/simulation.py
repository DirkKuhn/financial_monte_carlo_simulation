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
        self.factors_gen = _create_factors_gen(
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

            invest_increase, safe_increase, inflation_increase = self.factors_gen.sample(num_months)
            for j in range(num_months):  # Sequential
                # Update accumulators
                acc_invest = (acc_invest + monthly_invest) * invest_increase[j]
                acc_save = (acc_save + monthly_save) * safe_increase[j]
                acc_only_save = (acc_only_save + monthly_invest + monthly_save) * safe_increase[j]
                acc_inflation *= inflation_increase[j]

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


def _create_factors_gen(
        investment_return_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator,
        safe_deposit_rate_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator,
        inflation_rate_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator
) -> "FactorsGen":
    inv_values = safe_values = inflation_values = None

    if isinstance(investment_return_gen, HistoricalMonthlyGenerator):
        inv_values = investment_return_gen.values
    if isinstance(safe_deposit_rate_gen, HistoricalMonthlyGenerator):
        safe_values = safe_deposit_rate_gen.values
    if isinstance(inflation_rate_gen, HistoricalMonthlyGenerator):
        inflation_values = inflation_rate_gen.values

    values = [v for v in [inv_values, safe_values, inflation_values] if v is not None]
    if values:
        common_idx = values[0].index
        for i in range(1, len(values)):
            common_idx = common_idx.intersection(values[i].index)

    return FactorsGen(
        inv_values[common_idx] if inv_values is not None else investment_return_gen,
        safe_values[common_idx] if safe_values is not None else safe_deposit_rate_gen,
        inflation_values[common_idx] if inflation_values is not None else inflation_rate_gen
    )


@ti.data_oriented
class FactorsGen:
    def __init__(
            self,
            investment_returns: Sequence[float] | IndependentMonthlyGenerator,
            safe_rates: Sequence[float] | IndependentMonthlyGenerator,
            inflation_rates: Sequence[float] | IndependentMonthlyGenerator,
    ):
        self.ind_inv_ret = isinstance(investment_returns, IndependentMonthlyGenerator)
        self.ind_safe_rates = isinstance(safe_rates, IndependentMonthlyGenerator)
        self.ind_inflation_rates = isinstance(inflation_rates, IndependentMonthlyGenerator)

        self.investment_returns = investment_returns if self.ind_inv_ret else _convert(investment_returns)
        self.safe_rates = safe_rates if self.ind_safe_rates else _convert(safe_rates)
        self.inflation_rates = inflation_rates if self.ind_inflation_rates else _convert(inflation_rates)

        self.all_independent = self.ind_inv_ret and self.ind_safe_rates and self.ind_inflation_rates

    @ti.func
    def sample(self, num_months: int) -> tuple[ti.Field, ti.Field, ti.Field]:
        res_inv = self.investment_returns.sample_path(num_months) \
            if ti.static(self.ind_inv_ret) else ti.field(dtype=float, shape=num_months)
        res_safe = self.safe_rates.sample_path(num_months) \
            if ti.static(self.ind_safe_rates) else ti.field(dtype=float, shape=num_months)
        res_inflation = self.inflation_rates.sample_path(num_months) \
            if ti.static(self.inflation_rates) else ti.field(dtype=float, shape=num_months)

        if not self.all_independent:
            for i in range(num_months):
                idx = int(ti.floor(self.investment_returns.shape[0] * ti.random()))
                if self.ind_inv_ret:
                    res_inv[i] = self.investment_returns[idx]
                if self.ind_safe_rates:
                    res_safe[i] = self.safe_rates[idx]
                if self.ind_inflation_rates:
                    res_inflation[i] = self.inflation_rates[idx]

        return res_inv, res_safe, res_inflation


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
        num_sim=10_000,
        capital_gains_tax_rate=0.278186,
        investment_tax_exemption=0.3,
        investment_return_gen=HistoricalACWIIMIReturns(),
        safe_deposit_rate_gen=Historical1YearUSBondYields(),
        inflation_rate_gen=HistoricalGermanInflation()
    )

    sim.factors_gen.sample(12)

    from time import time

    for i in range(10):
        t = time()
        res = sim(
            num_years=500,
            current_invest=10_000, current_save=10_000,
            monthly_invest=1_000, monthly_save=1_000
        )
        print(f'Run {i}, duration: {time()-t}')
