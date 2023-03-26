import pandas as pd
import numpy as np


class TAQMetrics:
    def __init__(self, df):
        self.df = df

    def calculate_vwap(self):
        # Calculate the volume-weighted average price (VWAP)
        self.df['tv'] = self.df['Price'] * self.df['Volume']
        vwap = self.df['tv'].sum() / self.df['Volume'].sum()
        return vwap

    def calculate_vwap_sub(self):
        resampled_df = self.df.between_time("9:30", "15:30")
        resampled_df['tv'] = resampled_df['Price'] * resampled_df['Volume']
        vwap = resampled_df['tv'].sum() / resampled_df['Volume'].sum()
        return vwap

    def calculate_market_impact(self, vwap):
        # Calculate the permanent market impact (g)
        first_price = self.df.iloc[0]['Price']
        last_price = self.df.iloc[-1]['Price']
        g = last_price - first_price

        # Calculate the temporary market impact (h)
        h = vwap - first_price - g

        return g, h

    def calculate_imbalance(self):
        resampled_df = self.df.between_time("9:30", "15:30")
        resampled_df['Previous_price'] = resampled_df['Price'].shift(1)
        resampled_df.dropna(inplace=True)
        resampled_df['Trade_type'] = np.where(resampled_df['Price'] > resampled_df['Previous_price'],
                                              'buy', 'sell')

        # Initialize buy and sell volume sums
        buy_volume_sum = 0
        sell_volume_sum = 0

        # Iterate through the DataFrame and accumulate buy and sell volumes
        for index, row in resampled_df.iterrows():
            if row['Trade_type'] == 'buy':
                buy_volume_sum += row['Volume']
            else:
                sell_volume_sum += row['Volume']

        # Calculate imbalance
        imbalance = buy_volume_sum - sell_volume_sum

        return imbalance

    def calculate_mid_quote_returns_std(self):
        # Resample the dataframe to 2-minute intervals and take the first and last mid-quote prices
        resample_df = self.df.resample('2T')['midQuote'].agg({'first', 'last'})

        # Calculate the returns as the ratio of the last mid-quote price to the first mid-quote price, minus 1
        resample_df['return'] = resample_df['last'] / resample_df['first'] - 1

        # Calculate the standard deviation of the returns and scale it by the square root of the number of 2-minute
        # intervals in a trading day
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
