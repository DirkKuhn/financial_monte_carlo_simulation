from typing import NamedTuple

import numpy as np
from numba import njit

from generator import MonthlyReturnGenerator, FixedReturns, HistoricalReturns


class SimulationResult(NamedTuple):
    num_years: int
    current_invest: float
    current_save: float
    monthly_invest: float
    monthly_save: float
    total_value: np.ndarray
    value_only_safe_deposit: np.ndarray


class MonteCarloSimulation:
    def __init__(
            self,
            num_sim: int,
            interest_rate: float,
            capital_gains_tax_rate: float,
            investment_tax_exemption: float,
            yearly_inflation_rate: float,
            max_batch_size: int,
            investment_return_gen: MonthlyReturnGenerator = HistoricalReturns()
    ):
        self.num_sim = num_sim
        self.interest_rate = interest_rate
        self.capital_gains_tax_rate = capital_gains_tax_rate
        self.investment_tax_exemption = investment_tax_exemption
        self.yearly_inflation_rate = yearly_inflation_rate
        self.max_batch_size = max_batch_size
        self.investment_return_gen = investment_return_gen

    def simulate(
            self, num_years: int,
            current_invest: float, current_save: float,
            monthly_invest: float, monthly_save: float
    ) -> SimulationResult:
        """
        :param num_years:
        :param current_invest:
        :param current_save:
        :param monthly_invest:
        :param monthly_save:
        :return: Contains: total_value: [num_sim], value_only_safe_deposit: [num_sim]
        """
        num_months = 12 * num_years

        # [num_sim, num_months-1]
        monthly_returns = self.investment_return_gen.sample(num_rows=self.num_sim, num_months=num_months-1)
        monthly_interest = FixedReturns(self.interest_rate).sample(num_rows=self.num_sim, num_months=num_months-1)
        monthly_inflation = ...

        total_value = _simulate_after_tax_and_inflation(
            monthly_returns=monthly_returns, monthly_interest=monthly_interest, monthly_inflation=monthly_inflation,
            tax_exemption=self.investment_tax_exemption, capital_gains_tax_rate=self.capital_gains_tax_rate,
            current_invest=current_invest, current_save=current_save,
            monthly_invest=monthly_invest, monthly_save=monthly_save
        )

        value_only_safe_deposit = _simulate_after_tax_and_inflation(
            monthly_returns=monthly_returns, monthly_interest=monthly_interest, monthly_inflation=monthly_inflation,
            tax_exemption=self.investment_tax_exemption, capital_gains_tax_rate=self.capital_gains_tax_rate,
            current_invest=0, current_save=current_invest+current_save,
            monthly_invest=0, monthly_save=monthly_invest+monthly_save
        )

        return SimulationResult(
            num_years=num_years,
            current_invest=current_invest,
            current_save=current_save,
            monthly_invest=monthly_invest,
            monthly_save=monthly_save,
            total_value=total_value,
            value_only_safe_deposit=value_only_safe_deposit
        )


def _simulate_after_tax_and_inflation(
        monthly_returns: np.ndarray, monthly_interest: np.ndarray, monthly_inflation: np.ndarray,
        tax_exemption: float, capital_gains_tax_rate: float,
        current_invest: float, current_save: float, monthly_invest: float, monthly_save: float
) -> np.ndarray:
    investment_value_before_tax, total_investment = _simulate_before_tax(
        monthly_samples=monthly_returns, current_payment=current_invest, monthly_payment=monthly_invest
    )
    safe_deposit_value_before_tax, total_safe_deposited = _simulate_before_tax(
        monthly_samples=monthly_interest, current_payment=current_save, monthly_payment=monthly_save
    )

    # Total value before tax [num_sim]
    total_value = investment_value_before_tax + safe_deposit_value_before_tax

    # Total value after tax [num_sim]
    total_payment = total_investment + total_safe_deposited
    total_value -= _calc_capital_gains_tax(
        profit=total_value-total_payment, exemption=tax_exemption, rate=capital_gains_tax_rate
    )

    # Total value after tax and inflation [num_sim]
    total_value /= monthly_inflation.prod(axis=1)

    return total_value


def _simulate_before_tax(
        monthly_samples: np.ndarray, current_payment: float, monthly_payment: float
) -> tuple[np.ndarray, float]:
    num_months = monthly_samples.shape[1] + 1
    # [num_months]
    payments = np.full(shape=num_months, fill_value=monthly_payment)
    payments[0] = current_payment

    returns = _calc_returns(monthly_samples)  # [num_sim, num_months]
    value = (payments * returns).sum(axis=1)  # [num_sim]
    total_payment = payments.sum()

    return value, total_payment


def _calc_returns(monthly_returns: np.ndarray) -> np.ndarray:
    num_sim = monthly_returns.shape[0]
    # Apply ``cumprod`` backwards, i.e. the initial payment gets the most factors
    returns = monthly_returns[:, ::-1].cumprod(axis=1)[:, ::-1]
    return np.hstack((returns, np.ones((num_sim, 1))))


def _calc_capital_gains_tax(profit: np.ndarray, exemption: float, rate: float) -> np.ndarray:
    profit[profit < 0] = 0  # losses are not taxed
    return profit * (1-exemption) * rate