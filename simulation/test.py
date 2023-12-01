import pytest
from pytest import approx

from simulation import MonteCarloSimulation, FixedFactors
from simulation.run import calc_tax_invest, calc_tax_save, calc_tax_only_save


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
) -> tuple[float, float, float, float]:
    monthly_return = (1. + fixed_return) ** (1 / 12)
    monthly_rate = (1. + fixed_safe_deposit_rate) ** (1 / 12)

    expected_value_invest = (
            current_invest * monthly_return ** num_months
            + monthly_invest * ((1 - monthly_return ** (num_months + 1)) / (1 - monthly_return) - 1)
    )
    expected_value_save = (
        current_save * monthly_rate ** num_months
        + monthly_save * ((1 - monthly_rate ** (num_months + 1)) / (1 - monthly_rate) - 1)
    )
    expected_value = expected_value_invest + expected_value_save

    expected_value_only_safe_deposit = (
        (current_invest + current_save) * monthly_rate ** num_months
        + (monthly_invest + monthly_save) * ((1 - monthly_rate ** (num_months + 1)) / (1 - monthly_rate) - 1)
    )

    return expected_value, expected_value_only_safe_deposit, expected_value_invest, expected_value_save


def calc_expected_after_tax(
        investment_tax_exemption: float, capital_gains_tax_rate: float,
        value_invest: float, value_save: float, value_only_safe_deposit: float,
        current_invest: float, current_save: float,
        monthly_invest: float, monthly_save: float,
        num_months: int
) -> tuple[float, float]:
    invest_payment = current_invest + num_months * monthly_invest
    save_payment = current_save + num_months * monthly_save

    profit_invest = value_invest - invest_payment
    profit_save = value_save - save_payment
    profit_only_save = value_only_safe_deposit - (invest_payment + save_payment)

    tax_invest = calc_tax_invest(
        profit_invest=profit_invest, profit_save=profit_save,
        investment_tax_exemption=investment_tax_exemption, capital_gains_tax_rate=capital_gains_tax_rate
    )
    tax_save = calc_tax_save(
        profit_invest=profit_invest, profit_save=profit_save, capital_gains_tax_rate=capital_gains_tax_rate
    )
    tax_only_save = calc_tax_only_save(
        profit_only_save=profit_only_save, capital_gains_tax_rate=capital_gains_tax_rate
    )

    expected_value = value_invest + value_save - (tax_invest + tax_save)
    expected_value_only_save = value_only_safe_deposit - tax_only_save

    return expected_value, expected_value_only_save


@pytest.mark.parametrize(
    'fixed_return,fixed_safe_deposit_rate,current_invest,current_save,monthly_invest,monthly_save,num_months',
    [
        (0.04, 0.02, 10_000, 10_000, 1_000, 1_000, 12),
        (0.04, 0.02, 10_000, 10_000, 1_000, 1_000,  0),
        (0.04, 0.02, 10_000,      0, 1_000,     0, 12)
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
    expected_value, expected_value_only_safe_deposit, *_ = calc_expected_before_tax_and_inflation(
        fixed_return=fixed_return, fixed_safe_deposit_rate=fixed_safe_deposit_rate,
        current_invest=current_invest, current_save=current_save,
        monthly_invest=monthly_invest, monthly_save=monthly_save,
        num_months=num_months
    )
    assert value == approx(expected_value)
    assert value_only_safe_deposit == approx(expected_value_only_safe_deposit)


@pytest.mark.parametrize(
    'capital_gains_tax_rate,investment_tax_exemption,fixed_return,fixed_safe_deposit_rate,'
    'current_invest,current_save,monthly_invest,monthly_save,num_months',
    [
        (0.3, 0.2, 0.04, 0.02, 10_000, 10_000, 1_000, 1_000, 12),
        (0.3, 0.2, 0.04, 0.02, 10_000, 10_000, 1_000, 1_000, 0),
        (0.3, 0.2, 0.04, 0.02, 10_000,      0, 1_000,     0, 12)
    ]
)
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
    _, expected_value_only_safe_deposit, expected_value_invest, expected_value_save = calc_expected_before_tax_and_inflation(
        fixed_return=fixed_return, fixed_safe_deposit_rate=fixed_safe_deposit_rate,
        current_invest=current_invest, current_save=current_save,
        monthly_invest=monthly_invest, monthly_save=monthly_save,
        num_months=num_months
    )
    expected_value, expected_value_only_safe_deposit = calc_expected_after_tax(
        investment_tax_exemption=investment_tax_exemption, capital_gains_tax_rate=capital_gains_tax_rate,
        value_invest=expected_value_invest, value_save=expected_value_save,
        value_only_safe_deposit=expected_value_only_safe_deposit,
        current_invest=current_invest, current_save=current_save, monthly_invest=monthly_invest,
        monthly_save=monthly_save, num_months=num_months
    )
    assert value == approx(expected_value)
    assert value_only_safe_deposit == approx(expected_value_only_safe_deposit)


@pytest.mark.parametrize(
    'profit_invest,profit_save,investment_tax_exemption,capital_gains_tax_rate,expected_tax',
    [
        (1_000, 1_000, 0.3, 0.2, 1_000*(1-0.3)*0.2),
        (1_000,  -500, 0.3, 0.2, (1_000-500)*(1-0.3)*0.2),
        (-500,  1_000, 0.3, 0.2, 0),
        (-500,   -500, 0.3, 0.2, 0)
    ]
)
def test_calc_tax_invest(
        profit_invest: float, profit_save: float,
        investment_tax_exemption: float, capital_gains_tax_rate: float,
        expected_tax: float
):
    tax = calc_tax_invest(
        profit_invest=profit_invest, profit_save=profit_save,
        investment_tax_exemption=investment_tax_exemption, capital_gains_tax_rate=capital_gains_tax_rate
    )
    assert tax == approx(expected_tax)


@pytest.mark.parametrize(
    'profit_invest,profit_save,capital_gains_tax_rate,expected_tax',
    [
        (1_000, 1_000, 0.2, 1_000*0.2),
        (-500,  1_000, 0.2, (1_000-500)*0.2),
        (1_000,  -500, 0.2, 0),
        (-500,   -500, 0.2, 0)
    ]
)
def test_calc_tax_save(
        profit_invest: float, profit_save: float,
        capital_gains_tax_rate: float,
        expected_tax: float
):
    tax = calc_tax_save(
        profit_invest=profit_invest, profit_save=profit_save,
        capital_gains_tax_rate=capital_gains_tax_rate
    )
    assert tax == approx(expected_tax)


@pytest.mark.parametrize(
    'profit_only_save,capital_gains_tax_rate,expected_tax',
    [
        (1_000, 0.2, 1_000*0.2),
        (-500,  0.2, 0)
    ]
)
def test_calc_tax_only_save(
        profit_only_save: float,
        capital_gains_tax_rate: float,
        expected_tax: float
):
    tax = calc_tax_only_save(
        profit_only_save=profit_only_save,
        capital_gains_tax_rate=capital_gains_tax_rate
    )
    assert tax == approx(expected_tax)


@pytest.mark.parametrize(
    'capital_gains_tax_rate,investment_tax_exemption,fixed_return,fixed_safe_deposit_rate,'
    'fixed_inflation_rate,current_invest,current_save,monthly_invest,monthly_save,num_months',
    [
        (0.3, 0.2, 0.04, 0.02, 0.03, 10_000, 10_000, 1_000, 1_000, 12),
        (0.3, 0.2, 0.04, 0.02, 0.03, 10_000, 10_000, 1_000, 1_000, 0),
        (0.3, 0.2, 0.04, 0.02, 0.03, 10_000,      0, 1_000,     0, 12)
    ]
)
def test_fixed_rate_after_tax_and_inflation(
        capital_gains_tax_rate: float, investment_tax_exemption: float,
        fixed_return: float, fixed_safe_deposit_rate: float, fixed_inflation_rate: float,
        current_invest: float, current_save: float,
        monthly_invest: float, monthly_save: float,
        num_months: int
):
    value, value_only_safe_deposit = run(
        capital_gains_tax_rate=capital_gains_tax_rate, investment_tax_exemption=investment_tax_exemption,
        fixed_return=fixed_return, fixed_safe_deposit_rate=fixed_safe_deposit_rate,
        inflation_rate=fixed_inflation_rate,
        current_invest=current_invest, current_save=current_save,
        monthly_invest=monthly_invest, monthly_save=monthly_save,
        num_months=num_months
    )
    _, expected_value_only_safe_deposit, expected_value_invest, expected_value_save = calc_expected_before_tax_and_inflation(
        fixed_return=fixed_return, fixed_safe_deposit_rate=fixed_safe_deposit_rate,
        current_invest=current_invest, current_save=current_save,
        monthly_invest=monthly_invest, monthly_save=monthly_save,
        num_months=num_months
    )
    expected_value, expected_value_only_safe_deposit = calc_expected_after_tax(
        investment_tax_exemption=investment_tax_exemption, capital_gains_tax_rate=capital_gains_tax_rate,
        value_invest=expected_value_invest, value_save=expected_value_save,
        value_only_safe_deposit=expected_value_only_safe_deposit,
        current_invest=current_invest, current_save=current_save, monthly_invest=monthly_invest,
        monthly_save=monthly_save, num_months=num_months
    )
    monthly_inflation = (1. + fixed_inflation_rate) ** (1 / 12)
    expected_value /= monthly_inflation ** num_months
    expected_value_only_safe_deposit /= monthly_inflation ** num_months

    assert value == approx(expected_value)
    assert value_only_safe_deposit == approx(expected_value_only_safe_deposit)
