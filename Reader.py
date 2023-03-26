import os
from collections import defaultdict

import numpy as np
import pandas as pd

from TAQMetrics import TAQMetrics
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
from QuotesReader import TAQQuotesReader
from TAQAdjust import TAQAdjust, get_factor, get_adjust_date
from TradesReader import TAQTradesReader

WORK_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORK_DIR, 'data')
QUOTE_DIR = os.path.join(DATA_DIR, 'quotes')
TRADE_DIR = os.path.join(DATA_DIR, 'trades')
SP_PATH = os.path.join(WORK_DIR, 'data', 's&p500.xlsx')
tickers = ['MS', 'AAPL', 'MSFT', 'AMZN', 'JPM']
factor_df = get_factor(SP_PATH)
split_df = get_adjust_date(SP_PATH)


def calculate_volume_imbalance(self, resampled_df):
    # Calculate the volume imbalance
    vi = (resampled_df['volume'].iloc[0] - resampled_df['volume'].iloc[-1]) / resampled_df['volume'].sum()
    return vi


def calculate_std_2min_returns(self, resampled_df):
    # Calculate the 2-minute returns
    returns = np.log(resampled_df['price']) - np.log(resampled_df['price'].shift(1))

    # Calculate the standard deviation of the 2-minute returns
    std_returns = returns.std()

    return std_returns


def calculate_average_daily_volume(self, resampled_df):
    # Calculate the average daily volume
    avg_daily_volume = resampled_df['volume'].mean()

    return avg_daily_volume


if __name__ == '__main__':
    vwap400 = defaultdict(list)
    vwap330 = defaultdict(list)
    terminal_price = defaultdict(list)
    arrival_price = defaultdict(list)
    date_list = []
    market_imbalance = defaultdict(list)
    total_volume = defaultdict(list)
    return_std = defaultdict(list)
    for root, dir, file in os.walk(QUOTE_DIR):
        for date in dir:
            date_list.append(date)
            for subroot, subdir, subfile in os.walk(os.path.join(root, date)):
                for f in subfile:

                    ticker = f.split('_quotes')[0]
                    if ticker not in tickers:
                        continue
                    q_reader = TAQQuotesReader(os.path.join(subroot, f))
                    q_df = q_reader.get_df(date, ticker)
                    t_reader = TAQTradesReader(os.path.join(TRADE_DIR, date, ticker + '_trades.binRT'))
                    t_df = t_reader.get_df(date)

                    df = pd.merge(q_df, t_df, on='Date')

                    adjust = TAQAdjust(df=df, factor_df=factor_df, split_df=split_df)
                    adjust.adjust()
                    df = adjust.get_df()

                    df['midQuote'] = (df['BidPrice'] + df['AskPrice']) / 2
                    df.set_index('Date', inplace=True)

                    model = TAQMetrics(df)
                    vwap400[ticker].append(model.calculate_vwap())
                    return_std[ticker].append(model.calculate_mid_quote_returns_std())
                    vwap330[ticker].append(model.calculate_vwap_sub())
                    terminal_price[ticker].append(model.get_terminal_price())
                    arrival_price[ticker].append(model.get_arrival_price())
                    market_imbalance[ticker].append(model.calculate_imbalance())
                    total_volume[ticker].append(model.calculate_total_daily_volume())

                    print(vwap400)

    vwap400_df = pd.DataFrame.from_dict(vwap400, index=date_list)
    vwap330_df = pd.DataFrame.from_dict(vwap330, index=date_list)
    terminal_price_df = pd.DataFrame.from_dict(terminal_price, index=date_list)
    arrival_price_df = pd.DataFrame.from_dict(arrival_price, index=date_list)
    market_imbalance_df = pd.DataFrame.from_dict(market_imbalance, index=date_list)
    total_volume_df = pd.DataFrame.from_dict(total_volume, index=date_list)
    return_std_df = pd.DataFrame.from_dict(return_std, index=date_list)
    vwap400_df.to_csv("vwap400.csv")
    vwap330_df.to_csv("vwap330.csv")
    terminal_price_df.to_csv("terminal_price.csv")
    arrival_price_df.to_csv("arrival_price.csv")
    market_imbalance_df.to_csv("market_imbalance.csv")
    total_volume_df.to_csv("total_volume.csv")
    return_std_df.to_csv("return_std.csv")
    print(vwap330_df)
