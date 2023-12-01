from typing import NamedTuple, Union

import numpy as np
import taichi as ti

from simulation.generator import HistoricalMonthlyGenerator, IndependentMonthlyGenerator
from simulation.factor_generator import FactorsGen
from simulation.plots import ResultPlotter


class SimulationResult(NamedTuple):
    num_years: int
    current_invest: float
    current_save: float
    monthly_invest: float
    monthly_save: float
    value: np.ndarray
    value_only_safe_deposit: np.ndarray


@ti.data_oriented
class MonteCarloSimulation:
    def __init__(
            self,
            num_sim: int,
            capital_gains_tax_rate: float,
            investment_tax_exemption: float,
            investment_return_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator,
            safe_deposit_rate_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator,
            inflation_rate_gen: HistoricalMonthlyGenerator | IndependentMonthlyGenerator,
            arch: Union['ti.cpu', 'ti.gpu'] = ti.gpu,
            result_plotter: None | ResultPlotter = ResultPlotter()
    ):
        ti.init(arch=arch)

        self.num_sim = num_sim
        self.capital_gains_tax_rate = capital_gains_tax_rate
        self.investment_tax_exemption = investment_tax_exemption
        self.factors_gen = FactorsGen(
            investment_return_gen, safe_deposit_rate_gen, inflation_rate_gen
        )
        self.result_plotter = result_plotter
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
        result = SimulationResult(
            num_years=num_years,
            current_invest=current_invest,
            current_save=current_save,
            monthly_invest=monthly_invest,
            monthly_save=monthly_save,
            value=values[:, 0],
            value_only_safe_deposit=values[:, 1]
        )
        if self.result_plotter:
            self.result_plotter.print_result(result)
        return result

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
            tax_invest = wrapped_calc_tax_invest(
                profit_invest=profit_invest, profit_save=profit_save,
                investment_tax_exemption=self.investment_tax_exemption,
                capital_gains_tax_rate=self.capital_gains_tax_rate
            )
            tax_save = wrapped_calc_tax_save(
                profit_invest=profit_invest, profit_save=profit_save,
                capital_gains_tax_rate=self.capital_gains_tax_rate
            )
            tax_only_save = wrapped_calc_tax_only_save(
                profit_only_save=profit_only_save,
                capital_gains_tax_rate=self.capital_gains_tax_rate
            )

            value -= tax_invest + tax_save
            value_only_safe -= tax_only_save

            # Portfolio value after inflation
            value /= acc_inflation
            value_only_safe /= acc_inflation

            # Store portfolio value
            self.values[i, 0] = value
            self.values[i, 1] = value_only_safe


def calc_tax_invest(
        profit_invest: float, profit_save: float,
        investment_tax_exemption: float, capital_gains_tax_rate: float
) -> float:
    return (
        max(0., profit_invest + min(0., profit_save))
        * (1. - investment_tax_exemption) * capital_gains_tax_rate
    )


def calc_tax_save(
        profit_invest: float, profit_save: float,
        capital_gains_tax_rate: float
) -> float:
    return max(0., profit_save + min(0., profit_invest)) * capital_gains_tax_rate


def calc_tax_only_save(
        profit_only_save: float, capital_gains_tax_rate: float
) -> float:
    return max(0., profit_only_save) * capital_gains_tax_rate


wrapped_calc_tax_invest = ti.func(calc_tax_invest)
wrapped_calc_tax_save = ti.func(calc_tax_save)
wrapped_calc_tax_only_save = ti.func(calc_tax_only_save)


if __name__ == '__main__':
    from generator import (
        HistoricalACWIIMIUSDReturns,
        HistoricalACWIIMIEURReturns,
        Historical1YearUSBondYields,
        HistoricalECBRate,
        HistoricalGermanInflation
    )

    # sim = MonteCarloSimulation(
    #     num_sim=300_000,
    #     capital_gains_tax_rate=0.278186,
    #     investment_tax_exemption=0.3,
    #     investment_return_gen=HistoricalACWIIMIReturns(),
    #     safe_deposit_rate_gen=Historical1YearUSBondYields(),
    #     inflation_rate_gen=HistoricalGermanInflation(),
    #     result_plotter=None
    # )

    sim = MonteCarloSimulation(
        num_sim=300_000,
        capital_gains_tax_rate=0.278186,
        investment_tax_exemption=0.3,
        investment_return_gen=HistoricalACWIIMIEURReturns(),
        safe_deposit_rate_gen=HistoricalECBRate(),
        inflation_rate_gen=HistoricalGermanInflation(),
        result_plotter=None
    )

    from time import time

    for i in range(5):
        t = time()
        sim(
            num_years=100,
            current_invest=10_000, current_save=10_000,
            monthly_invest=1_000, monthly_save=1_000
        )
        print(f'Run {i}, duration: {time()-t}')
