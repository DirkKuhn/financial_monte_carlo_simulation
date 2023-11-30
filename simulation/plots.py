from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from simulation.simulation import SimulationResult


class ResultPlotter:
    def __init__(
            self,
            result: SimulationResult,
            percentiles: Sequence[float] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
    ):
        self.result = result
        self.percentiles = percentiles

    def print_result(self) -> None:
        self.print_info()
        self.print_statistics(self.percentiles)
        self.print_histogram()

    def print_info(self) -> None:
        print(f"Investment horizon:            {self.result.num_years} years")
        print(f"Currently invested:            {self.result.current_invest:,.0f}")
        print(f"Currently risk free deposited: {self.result.current_save:,.0f}")
        print(f"Monthly invested:              {self.result.monthly_invest:,.0f}")
        print(f"Monthly risk free deposited:   {self.result.monthly_save:,.0f}")

    def print_statistics(self, percentiles: Sequence[float]) -> None:
        df = pd.DataFrame(self.result.value)
        pd.options.display.float_format = '{:,.2f}'.format
        quantiles = df.quantile(percentiles)
        quantiles.index.name = "Percentiles"
        quantiles.columns = ["Resulting Wealth"]
        if is_notebook():
            display(quantiles)
        else:
            print(quantiles.to_string())

        print(f'Average resulting wealth:                   {self.result.value.mean():,.0f}')
        print(f'Average resulting wealth only safe deposit: {self.result.value_only_safe_deposit.mean():,.0f}')

        fraction_worse_outcomes = (
            (self.result.value < self.result.value_only_safe_deposit).sum() / len(self.result.value) * 100
        )
        print(f'Fraction of Worse Outcomes compared to not investing: {fraction_worse_outcomes:.1f}%')

        loss_idx = self.result.value < self.result.value_only_safe_deposit
        conditional_mean_loss = (self.result.value[loss_idx] - self.result.value_only_safe_deposit[loss_idx]).mean()
        print(f'Conditional mean loss: {conditional_mean_loss:.1f}')

    def print_histogram(self) -> None:
        fig, ax = plt.subplots()
        ax.hist(self.result.value, bins='auto')
        ax.set_title("Histogram of investment outcomes")
        ax.set_ylabel("Amount of observations")
        ax.set_xlabel("Resulting wealth after tax and inflation")
        ax.vlines(x=self.result.value_only_safe_deposit.mean(), ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='r')
        plt.tight_layout()
        if not is_notebook():
            plt.show()


def is_notebook() -> bool:
    """
    See: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
