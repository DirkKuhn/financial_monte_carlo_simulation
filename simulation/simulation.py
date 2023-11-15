from typing import NamedTuple

import numpy as np
from math import ceil

from generator import MonthlyReturnGenerator, FixedReturns, HistoricalReturns


class SimulationResult(NamedTuple):
    num_years: int
    current_invest: float
    current_save: float
    monthly_invest: float
    monthly_save: float
    annualized_return: float
    annualized_volatility: float
    is_sequence: bool
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
            monthly_invest: float, monthly_save: float,
            calculate_sequence: bool = False
    ) -> SimulationResult:
        """
        :param num_years:
        :param current_invest:
        :param current_save:
        :param monthly_invest:
        :param monthly_save:
        :param calculate_sequence:
        :return: Contains: total_value: [num_sim, num_months], value_only_safe_deposit: [num_months]
        """
        num_months = 12 * num_years
        safe_deposit_return_gen = FixedReturns(annualized_return=self.interest_rate)

        investment_value_after_tax = self._simulate_after_tax(
            num_rows=self.num_sim, num_months=num_months, return_gen=self.investment_return_gen,
            current_payment=current_invest, monthly_payment=monthly_invest,
            tax_exemption=self.investment_tax_exemption, calculate_sequence=calculate_sequence
        )  # [num_sim, num_months or 1]
        safe_deposit_value_after_tax = self._simulate_after_tax(
            num_rows=1, num_months=num_months, return_gen=safe_deposit_return_gen,
            current_payment=current_save, monthly_payment=monthly_save,
            tax_exemption=0, calculate_sequence=calculate_sequence
        )  # [1, num_months or 1]
        total_value = investment_value_after_tax + safe_deposit_value_after_tax  # [num_sim, num_months or 1]
        total_value = adjust_for_inflation(total_value, num_months, calculate_sequence, self.yearly_inflation_rate)

        value_only_safe_deposit = self._simulate_after_tax(
            num_rows=1, num_months=num_months, return_gen=safe_deposit_return_gen,
            current_payment=current_invest+current_save, monthly_payment=monthly_invest+monthly_save,
            tax_exemption=0, calculate_sequence=calculate_sequence
        )  # [1, num_months or 1]
        value_only_safe_deposit = adjust_for_inflation(value_only_safe_deposit, num_months, calculate_sequence, self.yearly_inflation_rate)
        value_only_safe_deposit = value_only_safe_deposit.squeeze(axis=0)  # [num_months or 1]

        return SimulationResult(
            num_years=num_years,
            current_invest=current_invest,
            current_save=current_save,
            monthly_invest=monthly_invest,
            monthly_save=monthly_save,
            annualized_return=self.investment_return_gen.annualized_return,
            annualized_volatility=self.investment_return_gen.annualized_volatility,
            is_sequence=calculate_sequence,
            total_value=total_value,
            value_only_safe_deposit=value_only_safe_deposit
        )

    def _simulate_after_tax_batched(
            self, num_rows: int, num_months: int, return_gen: MonthlyReturnGenerator,
            current_payment: float, monthly_payment: float, tax_exemption: float,
            calculate_sequence: bool
    ) -> np.ndarray:
        num_runs = ceil(num_rows / self.max_batch_size)
        num_rows_list = [(num_rows+i) // num_runs for i in range(num_runs)]
        value_list = []
        for num_rows in num_rows_list:
            values = self._simulate_after_tax(
                num_rows, num_months, return_gen, current_payment, monthly_payment, tax_exemption, calculate_sequence
            )
            value_list.append(values)
        return np.concatenate(value_list, axis=0)

    def _simulate_after_tax(
            self, num_rows: int, num_months: int, return_gen: MonthlyReturnGenerator,
            current_payment: float, monthly_payment: float, tax_exemption: float,
            calculate_sequence: bool
    ) -> np.ndarray:
        # [num_months]
        payments = np.full(shape=num_months, fill_value=monthly_payment)
        payments[0] = current_payment

        # [num_sim, num_months-1]
        monthly_returns = return_gen.sample(num_rows=num_rows, num_months=num_months - 1)

        if calculate_sequence:
            # [num_sim, num_months, num_months]
            return_matrix = sample_return_matrix(monthly_returns)

            # Value over time before tax and inflation
            value = (payments * return_matrix).sum(axis=-1)  # [num_sim, num_months]

            total_paid = payments.cumsum()  # [num_months]
        else:
            # [num_sim, num_months]
            returns = sample_returns(monthly_returns)

            # End value before tax and inflation
            value = (payments * returns).sum(axis=-1, keepdims=True)  # [num_sim, 1]

            total_paid = payments.sum()  # float

        # Value after tax and before inflation
        # [num_sim, num_months or 1]
        value -= calc_capital_gains_tax(
            profit=value-total_paid, exemption=tax_exemption, capital_gains_tax_rate=self.capital_gains_tax_rate
        )

        return value


def sample_return_matrix(monthly_returns: np.ndarray) -> np.ndarray:
    """
    :param monthly_returns: [num_sim, num_months-1]
    :return: return_matrix: [num_sim, num_months, num_months]
        [i,:,:]: [1,       0,       ..., 0
                  r1,      1,       ..., 0
                  r1*r2,   r2,      ..., 0
                  ...,              ...,
                  r1...rd, r2...rd, ..., 1]
    """
    num_sim = monthly_returns.shape[0]
    num_months = monthly_returns.shape[1] + 1
    return_matrix = np.tile(np.eye(num_months), reps=(num_sim, 1, 1))
    for i in range(num_months-1):
        return_matrix[:, (i + 1):, i] = monthly_returns[:, i:].cumprod(axis=1)
    return return_matrix


def sample_returns(monthly_returns: np.ndarray) -> np.ndarray:
    """
    :param monthly_returns: [num_sim, num_months-1]
    :return: returns: [num_sim, num_months]
    """
    num_sim = monthly_returns.shape[0]
    return_matrix = monthly_returns.cumprod(axis=1)[:, ::-1]  # [num_sim, num_months-1]
    return np.hstack([return_matrix, np.ones((num_sim, 1))])


def calc_capital_gains_tax(
        profit: float | np.ndarray, exemption: float, capital_gains_tax_rate: float
) -> float | np.ndarray:
    profit[profit < 0] = 0  # losses are not taxed
    return profit * (1-exemption) * capital_gains_tax_rate


def adjust_for_inflation(
        values: np.ndarray, num_months: int, calculate_sequence: bool, yearly_inflation_rate: float
) -> np.ndarray:
    monthly_factor_inflation = 1 / (1 + yearly_inflation_rate) ** (1 / 12)
    if calculate_sequence:
        inflation_factor = monthly_factor_inflation ** np.arange(num_months)  # [num_months]
    else:
        inflation_factor = monthly_factor_inflation ** (num_months - 1)  # float
    return values * inflation_factor  # [num_sim, num_months or 1]
