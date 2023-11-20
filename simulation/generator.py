import locale
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class HistoricalMonthlyGenerator(ABC):
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


class HistoricalACWIIMIReturns(HistoricalMonthlyGenerator):
    def __init__(self):
        world = self._read_file('data/world_historyIndex.xls')
        acwi_imi = self._read_file('data/acwi_imi_historyIndex.xls')

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

    @staticmethod
    def _read_file(path: str) -> pd.Series:
        return pd.read_excel(
            path, header=6, names=['date', 'value'], skipfooter=19,
            parse_dates=['date'], thousands=',', index_col='date', date_format="%b %d, %Y"
        ).squeeze()


class Historical1YearUSBondYields(HistoricalMonthlyGenerator):
    def __init__(self):
        yields_each_day = pd.read_csv("data/1-year-treasury-rate-yield-chart.csv", skiprows=15, index_col=0, parse_dates=True)
        yields_each_day = yields_each_day.dropna().squeeze()
        yields_each_day = yields_each_day / 100 + 1
        avg_yields_each_month = yields_each_day.resample('M').apply(lambda s: s.prod() ** (1/len(s))).to_period('M')
        approx_monthly_yields = avg_yields_each_month ** (1 / 12)
        self._values = approx_monthly_yields

        self.print_statistics()


class HistoricalGermanInflation(HistoricalMonthlyGenerator):
    def __init__(self):
        yearly_percentages_before_jan_1991 = pd.read_excel("data/API_FP.CPI.TOTL.ZG_DS2_en_excel_v2_5994828.xls", header=3, index_col=0)
        yearly_percentages_before_jan_1991 = yearly_percentages_before_jan_1991.loc['Germany'].iloc[3:]
        yearly_percentages_before_jan_1991 = yearly_percentages_before_jan_1991.astype(float) / 100 + 1
        yearly_percentages_before_jan_1991.index = pd.to_datetime(yearly_percentages_before_jan_1991.index)
        monthly_percentages_before_jan_1991 = yearly_percentages_before_jan_1991.resample('MS').ffill() ** (1 / 12)

        default_locale = locale.getlocale()
        locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")
        monthly_price_levels_after_jan_1991 = pd.read_excel(
            "data/61111-0002_$F.xlsx", header=3, index_col=[0], parse_dates=[[0, 1]],
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


class IndependentMonthlyGenerator(ABC):
    @abstractmethod
    def sample_path(self, num_months: int) -> np.ndarray:
        """
        Generate a single monthly return
        """


class FixedFactors(IndependentMonthlyGenerator):
    def __init__(self, annualized_increase: float):
        annualized_increase += 1
        self.monthly_factor = annualized_increase ** (1 / 12)

    def sample_path(self, num_months: int) -> np.ndarray:
        return np.full(num_months, fill_value=self.monthly_factor)

    @property
    def annualized_return(self) -> float:
        return self.monthly_factor ** 12

    @property
    def annualized_volatility(self) -> float:
        return 0


if __name__ == '__main__':
    gen = HistoricalACWIIMIReturns()
    gen = Historical1YearUSBondYields()
    gen = HistoricalGermanInflation()
