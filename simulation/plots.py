from collections.abc import Sequence

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from simulation import SimulationResult


class ResultPlotter:
    def __init__(self, result: SimulationResult):
        self.result = result

    @property
    def outcomes(self) -> np.ndarray:
        return self.result.total_value[:, -1]

    @property
    def outcome_only_safe_deposit(self) -> float:
        return self.result.value_only_safe_deposit[-1]

    def print_result(self, percentiles: Sequence[float] = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9)) -> None:
        self.print_info()
        self.print_statistics(percentiles)
        self.print_histogram()

    def print_info(self) -> None:
        print(f"Investment horizon:            {self.result.num_years} years")
        print(f"Currently invested:            {self.result.current_invest:,.0f}")
        print(f"Currently risk free deposited: {self.result.current_save:,.0f}")
        print(f"Monthly invested:              {self.result.monthly_invest:,.0f}")
        print(f"Monthly risk free deposited:   {self.result.monthly_save:,.0f}")
        print(f"Annualized return:             {self.result.annualized_return:.1f}%")
        print(f"Annualized volatility:         {self.result.annualized_volatility:.1f}%")

    def print_statistics(self, percentiles: Sequence[float]) -> None:
        df = pd.DataFrame(self.outcomes)
        pd.options.display.float_format = '{:,.2f}'.format
        quantiles = df.quantile(percentiles)
        quantiles.index.name = "Percentiles"
        quantiles.columns = ["Resulting Wealth"]
        display(quantiles)

        print(f'Average resulting wealth: {self.outcomes.mean():,.0f}')
        print(f'Only safe deposit:        {self.outcome_only_safe_deposit:,.0f}')

        fraction_worse_outcomes = (self.outcomes < self.outcome_only_safe_deposit).sum() / len(self.outcomes) * 100
        print(f'Fraction of Worse Outcomes compared to not investing: {fraction_worse_outcomes:.1f}%')
        
        conditional_mean_loss = self.outcomes[self.outcomes < self.outcome_only_safe_deposit].mean() \
                                - self.outcome_only_safe_deposit
        print(f'Conditional mean loss: {conditional_mean_loss:.1f}')

    def print_histogram(self) -> None:
        fig, ax = plt.subplots()
        ax.hist(self.outcomes, bins='auto')
        ax.set_title("Histogram of investment outcomes")
        ax.set_ylabel("Amount of observations")
        ax.set_xlabel("Resulting wealth after tax and inflation")
        ax.vlines(x=self.outcome_only_safe_deposit, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='r')
        plt.tight_layout()
        fig.show()

    def print_chart(self) -> None:
        assert self.result.is_sequence, "All values in the sequence must be calculated."
        fig, ax = plt.subplots()
        months = np.arange(0, self.result.num_years*12)
        no_invest, = ax.plot(months, self.result.value_only_safe_deposit, label='no investment', color='r')
        for values in self.result.total_value:
            invest, = ax.plot(months, values, label='simulation', color='b')
        ax.set_xlabel("Time in months")
        ax.set_ylabel("Value after tax and inflation")
        ax.set_title("Single simulations")
        ax.legend(handles=[no_invest, invest])
        plt.tight_layout()
        fig.show()
