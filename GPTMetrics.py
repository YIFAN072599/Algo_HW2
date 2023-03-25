import pandas as pd
import numpy as np


class TAQMetrics:
    def __init__(self, df):
        self.df = df

    def calculate_vwap(self):
        self.df['tv'] = self.df['Price'] * self.df['Volume']
        vwap = self.df['tv'].sum() / self.df['Volume'].sum()
        return vwap

    def calculate_vwap_sub(self):
        resampled_df = self.df.between_time("9:30", "15:30")
        resampled_df['tv'] = resampled_df['Price'] * resampled_df['Volume']
        vwap = resampled_df['tv'].sum() / resampled_df['Volume'].sum()
        return vwap

    def calculate_market_impact(self, vwap):
        first_price = self.df.iloc[0]['Price']
        last_price = self.df.iloc[-1]['Price']
        g = last_price - first_price
        h = vwap - first_price - g
        return g, h

    def calculate_imbalance(self):
        resampled_df = self.df.between_time("9:30", "15:30")
        resampled_df['Previous_price'] = resampled_df['Price'].shift(1)
        resampled_df.dropna(inplace=True)
        resampled_df['Trade_type'] = np.where(resampled_df['Price'] > resampled_df['Previous_price'], 'buy', 'sell')

        # Calculate buy and sell volume sums using vectorized operations
        buy_volume_sum = resampled_df.loc[resampled_df['Trade_type'] == 'buy', 'Volume'].sum()
        sell_volume_sum = resampled_df.loc[resampled_df['Trade_type'] == 'sell', 'Volume'].sum()

        # Calculate imbalance
        imbalance = buy_volume_sum - sell_volume_sum

        return imbalance

    def calculate_mid_quote_returns_std(self):
        resample_df = self.df.resample('2T')['midQuote'].agg(['first', 'last'])
        resample_df['return'] = resample_df['last'] / resample_df['first'] - 1
        return resample_df['return'].std() * np.sqrt(6.5 * 60 * 60 / 2)

    def calculate_total_daily_volume(self):
        total_volume = self.df["Volume"].sum()
        return total_volume

    def get_terminal_price(self):
        terminal_price = self.df[-5:]['midQuote'].mean()
        return terminal_price

    def get_arrival_price(self):
        arrival_price = self.df[:5]['midQuote'].mean()
        return arrival_price
