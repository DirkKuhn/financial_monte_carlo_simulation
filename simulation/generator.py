import locale
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import taichi as ti


parent_path = Path(__file__).parent


class HistoricalMonthlyGenerator(ABC):
    """
    Implement this class by providing time series data and storing it under ``_values``.
    The data is expected to be a ``pd.Series`` object of monthly factors.
    """
    _values: pd.Series

    def print_statistics(self) -> None:
        print(self.__class__.__name__)
        print(f'  Annualized increase:   {self.annualized_increase:4.1f}%')
        print(f'  Annualized volatility: {self.annualized_volatility:4.1f}%')
        print()

    @property
    def annualized_increase(self) -> float:
        mean_monthly_factor = self._values.prod() ** (1 / len(self._values))
        return (mean_monthly_factor ** 12 - 1) * 100

    @property
    def annualized_volatility(self) -> float:
        return np.std(np.log(self._values), ddof=1) * np.sqrt(12) * 100

    @property
    def values(self) -> pd.Series:
        return self._values


class HistoricalACWIIMIUSDReturns(HistoricalMonthlyGenerator):
    def __init__(self):
        world = read_msci_file('data/world_index_usd.xls')
        acwi_imi = read_msci_file('data/acwi_imi_index_usd.xls')

        assert world.index.min() < acwi_imi.index.min()

        first_date_acwi_imi = acwi_imi.index.min()
        value_world_at_first_date_acwi_imi = world.loc[first_date_acwi_imi]
        scaled_acwi_imi = acwi_imi / acwi_imi.loc[acwi_imi.index.min()] * value_world_at_first_date_acwi_imi

        world_before_acwi_imi = world[world.index < first_date_acwi_imi]
        absolute_values = pd.concat([world_before_acwi_imi, scaled_acwi_imi])

        # Check continuity
        assert not absolute_values.asfreq('bm').isnull().to_numpy().any()

        self._values = absolute_values.div(absolute_values.shift(1)).dropna()
        self._values.index = self._values.index.to_period('M')

        self.print_statistics()


class HistoricalACWIIMIEURReturns(HistoricalMonthlyGenerator):
    def __init__(self):
        acwi_imi = read_msci_file('data/acwi_imi_index_eur.xls')
        self._values = acwi_imi.div(acwi_imi.shift(1)).dropna()
        self._values.index = self._values.index.to_period('M')

        self.print_statistics()


def read_msci_file(path: str) -> pd.Series:
    return pd.read_excel(
        parent_path / path, header=6, names=['date', 'value'], skipfooter=19,
        parse_dates=['date'], thousands=',', index_col='date', date_format="%b %d, %Y"
    ).squeeze()


class Historical1YearUSBondYields(HistoricalMonthlyGenerator):
    def __init__(self):
        yields_each_day = pd.read_csv(
            parent_path / "data/1-year-treasury-rate-yield-chart.csv",
            skiprows=15, index_col=0, parse_dates=True
        )
        yields_each_day = yields_each_day.dropna().squeeze()
        yields_each_day = yields_each_day / 100 + 1
        avg_yields_each_month = yields_each_day.resample('M').apply(lambda s: s.prod() ** (1/len(s))).to_period('M')
        approx_monthly_yields = avg_yields_each_month ** (1 / 12)
        self._values = approx_monthly_yields

        self.print_statistics()


class HistoricalECBRate(HistoricalMonthlyGenerator):
    def __init__(self):
        ecb_rate = pd.read_excel(
            parent_path / "data/ecb_rate.xlsx", sheet_name="Daten",
            header=None, skiprows=5, index_col=1
        )
        ecb_rate = ecb_rate.iloc[:, 1]

        # Index to DateTime
        ecb_rate.index = ecb_rate.index.str.removeprefix("seit ")
        ecb_rate.index = ecb_rate.index.str.removesuffix("*")
        default_locale = locale.getlocale()
        locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")
        ecb_rate.index = pd.to_datetime(ecb_rate.index, format="%d. %B %Y")
        locale.setlocale(locale.LC_ALL, ".".join(default_locale))

        ecb_rate = ecb_rate / 100 + 1

        monthly_rate = ecb_rate.resample('MS').ffill() ** (1 / 12)

        last_idx = monthly_rate.index[-1]
        idx = pd.date_range(last_idx, pd.Timestamp.now(), freq="MS")
        monthly_rate_till_now = pd.Series(monthly_rate.iloc[-1], index=idx)

        self._values = pd.concat([
            monthly_rate.to_period('M'),
            monthly_rate_till_now.to_period('M').iloc[1:]]
        ).squeeze()

        self.print_statistics()


class HistoricalGermanInflation(HistoricalMonthlyGenerator):
    def __init__(self):
        yearly_percentages_before_jan_1991 = pd.read_excel(
            parent_path / "data/world_bank_yearly_inflation.xls", header=3, index_col=0
        )
        yearly_percentages_before_jan_1991 = yearly_percentages_before_jan_1991.loc['Germany'].iloc[3:]
        yearly_percentages_before_jan_1991 = yearly_percentages_before_jan_1991.astype(float) / 100 + 1
        yearly_percentages_before_jan_1991.index = pd.to_datetime(yearly_percentages_before_jan_1991.index)
        monthly_percentages_before_jan_1991 = yearly_percentages_before_jan_1991.resample('MS').ffill() ** (1 / 12)

        default_locale = locale.getlocale()
        locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")

        import warnings
        with warnings.catch_warnings(record=True):
            monthly_price_levels_after_jan_1991 = pd.read_excel(
                parent_path / "data/german_monthly_inflation.xlsx", header=3, index_col=[0], parse_dates=[[0, 1]],
                date_format="%Y %B", skiprows=[4], skipfooter=3, na_values="..."
            )
        locale.setlocale(locale.LC_ALL, ".".join(default_locale))
        monthly_price_levels_after_jan_1991 = monthly_price_levels_after_jan_1991.loc[:, "Verbraucherpreisindex"].dropna()
        monthly_percentages_after_jan_1991 = monthly_price_levels_after_jan_1991.shift(-1).div(monthly_price_levels_after_jan_1991).dropna()

        monthly_percentages_before_jan_1991 = monthly_percentages_before_jan_1991[
            monthly_percentages_before_jan_1991.index < monthly_percentages_after_jan_1991.index.min()
        ]
        monthly_percentages = pd.concat([monthly_percentages_before_jan_1991, monthly_percentages_after_jan_1991])
        self._values = monthly_percentages.to_period('M').squeeze()

        self.print_statistics()


@ti.data_oriented
class IndependentMonthlyGenerator(ABC):
    @abstractmethod
    def sample(self) -> float:
        """
        Generate a single monthly return
        """


class FixedFactors(IndependentMonthlyGenerator):
    def __init__(self, annualized_increase: float):
        annualized_increase += 1
        self.monthly_factor = annualized_increase ** (1 / 12)

    @ti.func
    def sample(self) -> float:
        return self.monthly_factor

    @property
    def annualized_return(self) -> float:
        return self.monthly_factor ** 12

    @property
    def annualized_volatility(self) -> float:
        return 0


if __name__ == '__main__':
    gen = HistoricalACWIIMIUSDReturns()
    gen = HistoricalACWIIMIEURReturns()
    gen = Historical1YearUSBondYields()
    gen = HistoricalECBRate()
    gen = HistoricalGermanInflation()
