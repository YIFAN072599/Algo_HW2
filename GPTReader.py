import os
from collections import defaultdict
import pandas as pd
from GPTMetrics import TAQMetrics
import warnings

from GPTQuotesReader import TAQQuotesReader
from GPTAdjust import prepare_adjustment_data, adjust_taq_data
from GPTTradesReader import TAQTradesReader

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

WORK_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORK_DIR, 'data')
QUOTE_DIR = os.path.join(DATA_DIR, 'quotes')
TRADE_DIR = os.path.join(DATA_DIR, 'trades')
SP_PATH = os.path.join(WORK_DIR, 'data', 's&p500.xlsx')
tickers = ['MS', 'AAPL', 'MSFT', 'AMZN', 'JPM']
factor_df, split_df = prepare_adjustment_data()


class TAQAnalysis:
    def __init__(self, quote_dir, trade_dir, tickers, factor_df, split_df):
        self.quote_dir = quote_dir
        self.trade_dir = trade_dir
        self.tickers = tickers
        self.factor_df = factor_df
        self.split_df = split_df
        self.results = defaultdict(lambda: defaultdict(list))

    def run(self):
        for root, dir, files in os.walk(self.quote_dir):
            for date in dir[:1]:
                for subroot, subdir, subfiles in os.walk(os.path.join(root, date)):
                    for f in subfiles:

                        ticker = f.split('_quotes')[0]
                        if ticker not in self.tickers:
                            continue

                        q_reader = TAQQuotesReader(os.path.join(subroot, f))
                        q_df = q_reader.get_df(date, ticker)
                        t_reader = TAQTradesReader(os.path.join(self.trade_dir, date, ticker + '_trades.binRT'))
                        t_df = t_reader.get_df(date)

                        df = pd.merge(q_df, t_df, on='Date')

                        df = adjust_taq_data(df, self.factor_df, self.split_df)

                        df['midQuote'] = (df['BidPrice'] + df['AskPrice']) / 2
                        df.set_index('Date', inplace=True)

                        metric = TAQMetrics(df)
                        self.results[ticker]['vwap400'].append(metric.calculate_vwap())
                        self.results[ticker]['return_std'].append(metric.calculate_mid_quote_returns_std())
                        self.results[ticker]['vwap330'].append(metric.calculate_vwap_sub())
                        self.results[ticker]['terminal_price'].append(metric.get_terminal_price())
                        self.results[ticker]['arrival_price'].append(metric.get_arrival_price())
                        self.results[ticker]['market_imbalance'].append(metric.calculate_imbalance())
                        self.results[ticker]['total_volume'].append(metric.calculate_total_daily_volume())
                        self.results[ticker]['h'].append(metric.calculate_market_impact()[1])

    def save_results(self, file_prefix=''):
        for key, result in self.results.items():
            for metric, values in result.items():
                filename = f"{file_prefix}{metric}_{key}.csv"
                pd.DataFrame(values).to_csv(filename, index=False)


if __name__ == '__main__':
    analysis = TAQAnalysis(QUOTE_DIR, TRADE_DIR, tickers, factor_df, split_df)
    analysis.run()
    analysis.save_results()
