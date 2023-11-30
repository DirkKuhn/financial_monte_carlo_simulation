import pytest
from pytest import approx

from simulation import MonteCarloSimulation, FixedFactors


def run(
        capital_gains_tax_rate: float, investment_tax_exemption: float,
        fixed_return: float, fixed_safe_deposit_rate: float, inflation_rate: float,
        current_invest: float, current_save: float,
        monthly_invest: float, monthly_save: float,
        num_months: int
) -> tuple[float, float]:
    sim = MonteCarloSimulation(
        num_sim=1,
        capital_gains_tax_rate=capital_gains_tax_rate,
        investment_tax_exemption=investment_tax_exemption,
        investment_return_gen=FixedFactors(fixed_return),
        safe_deposit_rate_gen=FixedFactors(fixed_safe_deposit_rate),
        inflation_rate_gen=FixedFactors(inflation_rate)
    )
    sim._simulate(
        num_months=num_months,
        current_invest=current_invest, current_save=current_save,
        monthly_invest=monthly_invest, monthly_save=monthly_save
    )
    res = sim.values.to_numpy()

    value = res[0, 0]
    value_only_safe_deposit = res[0, 1]
    return value, value_only_safe_deposit


def calc_expected_before_tax_and_inflation(
        fixed_return: float, fixed_safe_deposit_rate: float,
        current_invest: float, current_save: float,
        monthly_invest: float, monthly_save: float,
        num_months: int
) -> tuple[float, float]:
    monthly_return = (1. + fixed_return) ** (1 / 12)
    monthly_rate = (1. + fixed_safe_deposit_rate) ** (1 / 12)

    expected_value = (
        current_invest * monthly_return ** num_months
        + monthly_invest * ((1 - monthly_return ** (num_months + 1)) / (1 - monthly_return) - 1)
        + current_save * monthly_rate ** num_months
        + monthly_save * ((1 - monthly_rate ** (num_months + 1)) / (1 - monthly_rate) - 1)
    )

    expected_value_only_safe_deposit = (
        (current_invest + current_save) * monthly_rate ** num_months
        + (monthly_invest + monthly_save) * ((1 - monthly_rate ** (num_months + 1)) / (1 - monthly_rate) - 1)
    )

    return expected_value, expected_value_only_safe_deposit


@pytest.mark.parametrize(
    'fixed_return,fixed_safe_deposit_rate,current_invest,current_save,monthly_invest,monthly_save,num_months',
    [
        (0.04, 0.02, 10_000, 10_000, 1_000, 1_000, 12),
        (0.04, 0.02, 10_000, 10_000, 1_000, 1_000,  0),
        (0.04, 0.02, 10_000,      0, 1_000,     0, 12),
    ]
)
def test_fixed_rate_before_tax_and_inflation(
        fixed_return: float, fixed_safe_deposit_rate: float,
        current_invest: float, current_save: float,
        monthly_invest: float, monthly_save: float,
        num_months: int
):
    value, value_only_safe_deposit = run(
        capital_gains_tax_rate=0., investment_tax_exemption=0.,
        fixed_return=fixed_return, fixed_safe_deposit_rate=fixed_safe_deposit_rate,
        inflation_rate=0.,
        current_invest=current_invest, current_save=current_save,
        monthly_invest=monthly_invest, monthly_save=monthly_save,
        num_months=num_months
    )
    expected_value, expected_value_only_safe_deposit = calc_expected_before_tax_and_inflation(
        fixed_return=fixed_return, fixed_safe_deposit_rate=fixed_safe_deposit_rate,
        current_invest=current_invest, current_save=current_save,
        monthly_invest=monthly_invest, monthly_save=monthly_save,
        num_months=num_months
    )
    assert value == approx(expected_value)
    assert value_only_safe_deposit == approx(expected_value_only_safe_deposit)


def test_fixed_rate_after_tax_and_before_inflation(
        capital_gains_tax_rate: float, investment_tax_exemption: float,
        fixed_return: float, fixed_safe_deposit_rate: float,
        current_invest: float, current_save: float,
        monthly_invest: float, monthly_save: float,
        num_months: int
):
    value, value_only_safe_deposit = run(
        capital_gains_tax_rate=capital_gains_tax_rate, investment_tax_exemption=investment_tax_exemption,
        fixed_return=fixed_return, fixed_safe_deposit_rate=fixed_safe_deposit_rate,
        inflation_rate=0.,
        current_invest=current_invest, current_save=current_save,
        monthly_invest=monthly_invest, monthly_save=monthly_save,
        num_months=num_months
    )
    expected_value, expected_value_only_safe_deposit = calc_expected_before_tax_and_inflation(
        fixed_return=fixed_return, fixed_safe_deposit_rate=fixed_safe_deposit_rate,
        current_invest=current_invest, current_save=current_save,
        monthly_invest=monthly_invest, monthly_save=monthly_save,
        num_months=num_months
    )
    assert value == approx(expected_value)
    assert value_only_safe_deposit == approx(expected_value_only_safe_deposit)


def test_fixed_rate_after_tax():
    tax_rate = 0.3
    fixed_rate = 0.04
    current_payment = 10_000
    monthly_payment = 100
    num_months = 20

    sim = MonteCarloSimulation(
        num_sim=0,
        interest_rate=0,
        capital_gains_tax_rate=tax_rate,
        investment_tax_exemption=0,
        yearly_inflation_rate=0,
        max_batch_size=0
    )
    result = sim._simulate_value_over_time_after_tax(
        num_rows=1, num_months=num_months, return_gen=FixedReturns(annualized_return=fixed_rate),
        current_payment=current_payment, monthly_payment=monthly_payment, tax_exemption=0,
        calculate_sequence=False
    )
    result_sequence = sim._simulate_value_over_time_after_tax(
        num_rows=1, num_months=num_months, return_gen=FixedReturns(annualized_return=fixed_rate),
        current_payment=current_payment, monthly_payment=monthly_payment, tax_exemption=0,
        calculate_sequence=True
    )

    monthly_factor = (1 + fixed_rate) ** (1 / 12)
    expected_result_before_tax = current_payment * monthly_factor ** (num_months - 1) \
        + monthly_payment * (1 - monthly_factor ** (num_months - 1)) / (1 - monthly_factor)
    expected_result = expected_result_before_tax \
        - (expected_result_before_tax - (current_payment + (num_months-1) * monthly_payment)) * tax_rate

    assert result[0, 0] == approx(expected_result), \
        f"Expected equality but got {result[0, 0]} and {expected_result}"
    assert result_sequence[0, -1] == approx(expected_result), \
        f"Expected equality but got {result_sequence[0, -1]} and {expected_result}"


def test_fixed_rate_after_tax_and_inflation():
    inflation_rate = 0.03
    tax_rate = 0.3
    fixed_rate = 0.04
    current_payment = 10_000
    monthly_payment = 100
    num_years = 10

    sim = MonteCarloSimulation(
        num_sim=1,
        interest_rate=fixed_rate,
        capital_gains_tax_rate=tax_rate,
        investment_tax_exemption=0,
        yearly_inflation_rate=inflation_rate,
        max_batch_size=1,
        investment_return_gen=FixedReturns(annualized_return=fixed_rate)
    )
    result = sim.simulate(
        num_years=num_years, current_invest=0, current_save=current_payment,
        monthly_invest=0, monthly_save=monthly_payment, calculate_sequence=False
    )
    result = result.value_only_safe_deposit[0]

    result_sequence = sim.simulate(
        num_years=num_years, current_invest=0, current_save=current_payment,
        monthly_invest=0, monthly_save=monthly_payment, calculate_sequence=True
    )
    result_sequence = result_sequence.value_only_safe_deposit[-1]

    result_invest = sim.simulate(
        num_years=num_years, current_invest=current_payment, current_save=0,
        monthly_invest=monthly_payment, monthly_save=0, calculate_sequence=False
    )
    result_invest = result_invest.total_value[0, 0]

    result_invest_sequence = sim.simulate(
        num_years=num_years, current_invest=current_payment, current_save=0,
        monthly_invest=monthly_payment, monthly_save=0, calculate_sequence=True
    )
    result_invest_sequence = result_invest_sequence.total_value[0, -1]

    num_months = num_years * 12
    monthly_factor = (1 + fixed_rate) ** (1 / 12)
    expected_result_before_tax = current_payment * monthly_factor ** (num_months - 1) \
        + monthly_payment * (1 - monthly_factor ** (num_months - 1)) / (1 - monthly_factor)
    expected_result_after_tax = expected_result_before_tax \
        - (expected_result_before_tax - (current_payment + (num_months-1) * monthly_payment)) * tax_rate
    monthly_factor_inflation = 1 / (1 + inflation_rate) ** (1 / 12)
    expected_result = expected_result_after_tax * monthly_factor_inflation ** (num_months - 1)

    assert result == approx(expected_result), \
        f"Expected equality but got {result} and {expected_result}"
    assert result_sequence == approx(expected_result), \
        f"Expected equality but got {result_sequence} and {expected_result}"
    assert result_invest == approx(expected_result), \
        f"Expected equality but got {result_invest} and {expected_result}"
    assert result_invest_sequence == approx(expected_result), \
        f"Expected equality but got {result_invest_sequence} and {expected_result}"
