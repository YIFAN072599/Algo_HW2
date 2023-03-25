import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

WORK_DIR = os.path.dirname(__file__)
CLEAN_QUOTE_DIR = os.path.join(WORK_DIR, 'data', 'quote')
CLEAN_TRADE_DIR = os.path.join(WORK_DIR, 'data', 'trade')
SP_PATH = os.path.join(WORK_DIR, 'data', 's&p500.xlsx')
FIG_DIR = os.path.join(WORK_DIR, 'figure', 'clean')


def remove_trade_outliers(trade_df, rolling_window, threshold_error):
    adj_price = trade_df['Adj_Price']
    mean = adj_price.rolling(rolling_window).mean()
    std = adj_price.rolling(rolling_window).std()
    error = threshold_error * adj_price.mean()
    upper_bound = mean + 2 * std + error
    lower_bound = mean - 2 * std - error
    mask = (adj_price > upper_bound) | (adj_price < lower_bound)
    mask.iloc[:rolling_window - 1] = False
    trade_df.loc[mask, ["Price", "Adj_Price", "Size", "Adj_Size"]] = np.nan
    return trade_df


class TAQCleaner(object):
    def __init__(self, quote_path, trade_path, rolling_window=25, threshold_error=0.0002):
        self._threshold_error = threshold_error
        self._rolling_window = rolling_window
        self._quote_path = quote_path
        self._trade_path = trade_path
        workdir = os.listdir(quote_path)
        if '.DS_Store' in workdir:
            workdir.remove('.DS_Store')
        self._ticker = [x[:-4] for x in workdir]

    def read_trade_df(self, ticker):
        return pd.read_csv(os.path.join(self._trade_path, ticker + '.csv'), parse_dates=['Date'], index_col='Date')

    def write_trade_df(self, trade_df, ticker):
        trade_df.to_csv(os.path.join(self._trade_path, ticker + '_clean.csv'))

    def read_quote_df(self, ticker):
        return pd.read_csv(os.path.join(self._quote_path, ticker + '.csv'), parse_dates=['Date'], index_col='Date')

    def write_quote_df(self, quote_df, ticker):
        quote_df.to_csv(os.path.join(self._quote_path, ticker + '_clean.csv'))

    def clean_trade(self):
        """
        Cleans the trade data by removing outliers.
        """
        for ticker in self._ticker:
            if ticker.endswith('clean'):
                continue
            trade_df = self.read_trade_df(ticker)
            trade_df = remove_trade_outliers(trade_df, self._rolling_window, self._threshold_error)
            self.write_trade_df(trade_df, ticker)

    def clean_quote(self):
        """
        Cleans the quote data by removing outliers.
        """
        for ticker in self._ticker:
            if ticker.endswith('clean'):
                continue
            quote_df = self.read_quote_df(ticker)
            adj_price_columns = ['Adj_AskPrice', 'Adj_BidPrice']
            adj_size_columns = ['Adj_AskSize', 'Adj_BidSize']
            for i, adj_price_col in enumerate(adj_price_columns):
                adj_price = quote_df[adj_price_col]
                mean = adj_price.rolling(self._rolling_window).mean()
                std = adj_price.rolling(self._rolling_window).std()
                error = self._threshold_error * adj_price.mean()
                upper_bound = mean + 2 * std + error
                lower_bound = mean - 2 * std - error
                mask = (adj_price > upper_bound) | (adj_price < lower_bound)
                mask.iloc[:self._rolling_window - 1] = False
                adj_size_col = adj_size_columns[i]
                quote_df.loc[mask, [adj_price_col.replace("Adj_", ""), adj_price_col, adj_size_col.replace("Adj_", ""),
                                    adj_size_col]] = np.nan
                quote_df['Mid_Price'] = (quote_df['Adj_AskPrice'] + quote_df['Adj_BidPrice'])/2
            self.write_quote_df(quote_df, ticker)

    def plot(self):
        if not os.path.exists(FIG_DIR):
            os.mkdir(FIG_DIR)
        plt.figure(figsize=(16, 9))
        for ticker in self._ticker:
            if not ticker.endswith('clean'):
                continue
            quote = pd.read_csv(os.path.join(self._quote_path, ticker + '.csv'), parse_dates=['Date'])
            plt.plot(quote['Date'], quote['Adj_AskPrice'], label=ticker + ' Adj_AskPrice')
            plt.plot(quote['Date'], quote['Adj_BidPrice'], label=ticker + ' Adj_BidPrice')
            plt.legend()
            plt.title(f'{ticker} Quote Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.savefig(os.path.join(FIG_DIR, f'{ticker}_quote_prices.png'))
            plt.clf()

        # Plot trade's Adj_Price in a separate figure
        plt.figure(figsize=(16, 9))
        for ticker in self._ticker:
            if not ticker.endswith('clean'):
                continue
            trade = pd.read_csv(os.path.join(self._trade_path, ticker + '.csv'), parse_dates=['Date'])
            plt.plot(trade['Date'], trade['Adj_Price'], label=ticker + ' Adj_Price')
            plt.legend()
            plt.title(f'{ticker} Trade Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.savefig(os.path.join(FIG_DIR, f'{ticker}_trade_prices.png'))
            plt.clf()


if __name__ == '__main__':
    cleaner = TAQCleaner(CLEAN_QUOTE_DIR, CLEAN_TRADE_DIR)
    #cleaner.clean_quote()
    #cleaner.clean_trade()
    cleaner.plot()
