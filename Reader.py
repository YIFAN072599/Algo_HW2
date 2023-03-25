import os

import pandas as pd

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

if __name__ == '__main__':
    for root, dir, file in os.walk(QUOTE_DIR):
        for date in dir:
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
                    resampled_df = df.resample('2T').agg(
                        {'Ticker': 'first', 'midQuote': 'mean', 'Price': 'mean', 'Volume': 'sum'})

                    print(resampled_df)
