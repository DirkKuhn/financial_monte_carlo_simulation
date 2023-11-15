from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class MonthlyReturnGenerator(ABC):
    @abstractmethod
    def sample(self, num_rows: int, num_months: int) -> np.ndarray:
        """
        Generate monthly returns
        :param num_rows:
        :param num_months:
        :return: [num_rows, num_months]
        """

    @property
    @abstractmethod
    def annualized_return(self) -> float:
        pass

    @property
    @abstractmethod
    def annualized_volatility(self) -> float:
        pass


class FixedReturns(MonthlyReturnGenerator):
    def __init__(self, annualized_return: float):
        annualized_return += 1
        self.monthly_return = annualized_return ** (1 / 12)

    def sample(self, num_rows: int, num_months: int) -> np.ndarray:
        return np.full(shape=(num_rows, num_months), fill_value=self.monthly_return)

    @property
    def annualized_return(self) -> float:
        return self.monthly_return ** 12

    @property
    def annualized_volatility(self) -> float:
        return 0


class HistoricalReturns(MonthlyReturnGenerator):
    def __init__(self):
        world = self._read_file('data/world_history.xls')
        acwi_imi = self._read_file('data/acwi_imi_history.xls')

        first_date_acwi_imi = acwi_imi.index[0]
        value_world_at_first_date_acwi_imi = world.loc[first_date_acwi_imi][0]
        scaled_acwi_imi = acwi_imi / acwi_imi.iloc[0, 0] * value_world_at_first_date_acwi_imi

        world_before_acwi_imi = world.loc[:first_date_acwi_imi].iloc[:-1]  # Slice also contains bounds
        # [num_returns, 1]
        hist_values = np.vstack((world_before_acwi_imi.to_numpy(), scaled_acwi_imi.to_numpy()))
        hist_values = hist_values.squeeze(axis=1)
        self.hist_returns = hist_values[1:] / hist_values[:-1]

    @staticmethod
    def _read_file(path: str) -> pd.DataFrame:
        return pd.read_excel(path, header=6, names=['date', 'value'], skipfooter=19,
                             parse_dates=[0], thousands=',', index_col=0)

    def sample(self, num_rows: int, num_months: int) -> np.ndarray:
        return np.random.choice(self.hist_returns, size=(num_rows, num_months))

    @property
    def annualized_return(self) -> float:
        num_monthly_returns = len(self.hist_returns)
        mean_monthly_factor = self.hist_returns.prod() ** (1 / num_monthly_returns)
        return (mean_monthly_factor ** 12 - 1) * 100

    @property
    def annualized_volatility(self) -> float:
        return np.std(np.log(self.hist_returns), ddof=1) * np.sqrt(12) * 100


if __name__ == '__main__':
    gen = HistoricalReturns()
    print(f'Annualized return:      {gen.annualized_return:.1f}%')
    print(f'Annualized volatility: {gen.annualized_volatility:.1f}%')
