from typing import TYPE_CHECKING
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from simulation.run import SimulationResult


class ResultPlotter:
    def __init__(
            self,
            percentiles: Sequence[float] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
    ):
        self.percentiles = percentiles

    def print_result(self, result: 'SimulationResult') -> None:
        self.print_statistics(result)
        self.print_histogram(result)

    def print_statistics(self, result: 'SimulationResult') -> None:
        df = pd.DataFrame(result.value)
        pd.options.display.float_format = '{:,.2f}'.format
        quantiles = df.quantile(self.percentiles)
        quantiles.index.name = "Percentiles"
        quantiles.columns = ["Resulting Wealth"]
        if is_notebook():
            display(quantiles)
        else:
            print(quantiles.to_string())

        print(f'Average resulting wealth:                   {result.value.mean():,.0f}')
        print(f'Average resulting wealth only safe deposit: {result.value_only_safe_deposit.mean():,.0f}')

        fraction_worse_outcomes = (
            (result.value < result.value_only_safe_deposit).sum() / len(result.value) * 100
        )
        print(f'Fraction of Worse Outcomes compared only safe deposit: {fraction_worse_outcomes:.1f}%')

        loss_idx = result.value < result.value_only_safe_deposit
        conditional_mean_loss = (result.value[loss_idx] - result.value_only_safe_deposit[loss_idx]).mean()
        print(f'Conditional mean loss: {conditional_mean_loss:.1f}')

        total_payment = result.current_invest + result.current_save
        total_payment += result.num_years * 12 * (result.monthly_invest + result.monthly_save)
        print(f'Total payment: {total_payment:.0f}')

    def print_histogram(self, result: 'SimulationResult') -> None:
        fig, ax = plt.subplots()
        ax.hist(result.value, bins='auto')
        ax.set_title("Histogram of investment outcomes")
        ax.set_ylabel("Amount of observations")
        ax.set_xlabel("Resulting wealth after tax and inflation")
        ax.vlines(x=result.value_only_safe_deposit.mean(), ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='r')
        plt.tight_layout()
        if not is_notebook():
            plt.show()


def is_notebook() -> bool:
    """
    Check whether this is run from a jupyter notebook or a terminal.
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
