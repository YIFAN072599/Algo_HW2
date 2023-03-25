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
        self.resampled_df = self.df.between_time("9:30", "15:30")
        self.resampled_df['Previous_price'] = self.resampled_df['Price'].shift(1)
        self.resampled_df.dropna(inplace=True)
        self.resampled_df['Trade_type'] = np.where(self.resampled_df['Price'] > self.resampled_df['Previous_price'],
                                                   'buy', 'sell')

        # Initialize buy and sell volume sums
        buy_volume_sum = 0
        sell_volume_sum = 0

        # Iterate through the DataFrame and accumulate buy and sell volumes
        for index, row in self.resampled_df.iterrows():
            if row['Trade_type'] == 'buy':
                buy_volume_sum += row['Volume']
            else:
                sell_volume_sum += row['Volume']

        # Calculate imbalance
        imbalance = buy_volume_sum - sell_volume_sum

        return imbalance

    def calculate_mid_quote_returns_std(self):
        resample_df = self.df["midQuote"]
        resampled_first = self.df.resample('2T').agg(
         {'Ticker': 'first', 'midQuote': 'mean', 'Price': 'mean', 'Volume': 'sum'})
        resampled_last = self.df.resample('2T').agg(
         {'Ticker': 'last', 'midQuote': 'mean', 'Price': 'mean', 'Volume': 'sum'})
        freq = "2t"
        return_df = resample_df(freq, closed="left", label="right", on="Date")
        first = return_df.agg("first")
        last = return_df.agg("last")
        mid_return_df = last[["midQuote"]] / first[["midQuote"]] - 1
        mid_return_df.rename(columns={"midQuote": freq + "_ret"}, inplace=True)
        return mid_return_df[freq + "_ret"].std() * np.sqrt(6.5 * 60 * 60 / 2)

    def calculate_total_daily_volume(self):
        total_volume = self.df["Volume"].sum()
        return total_volume

    def get_terminal_price(self):
        terminal_price = self.df[-5:]['midQuote'].mean()
        return terminal_price

    def get_arrival_price(self):
        arrival_price = self.df[:5]['midQuote'].mean()
        return arrival_price

