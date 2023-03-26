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
        self.vwap400 = defaultdict(list)
        self.vwap330 = defaultdict(list)
        self.terminal_price = defaultdict(list)
        self.arrival_price = defaultdict(list)
        self.date_list = []
        self.market_imbalance = defaultdict(list)
        self.total_volume = defaultdict(list)
        self.return_std = defaultdict(list)
        self.temporary_impact = defaultdict(list)

    def run(self):
        for root, dir, files in os.walk(self.quote_dir):
            for date in dir:
                self.date_list.append(date)
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
                        self.vwap400[ticker].append(metric.calculate_vwap())
                        self.return_std[ticker].append(metric.calculate_mid_quote_returns_std())
                        self.vwap330[ticker].append(metric.calculate_vwap_sub())
                        self.terminal_price[ticker].append(metric.get_terminal_price())
                        self.arrival_price[ticker].append(metric.get_arrival_price())
                        self.market_imbalance[ticker].append(metric.calculate_imbalance())
                        self.total_volume[ticker].append(metric.calculate_total_daily_volume())
                        self.temporary_impact[ticker].append(metric.calculate_market_impact()[1])

    def save_results(self):
        vwap400_df = pd.DataFrame(self.vwap400, index=self.date_list)
        vwap330_df = pd.DataFrame(self.vwap330, index=self.date_list)
        terminal_price_df = pd.DataFrame(self.terminal_price, index=self.date_list)
        arrival_price_df = pd.DataFrame(self.arrival_price, index=self.date_list)
        market_imbalance_df = pd.DataFrame(self.market_imbalance, index=self.date_list)
        total_volume_df = pd.DataFrame(self.total_volume, index=self.date_list)
        return_std_df = pd.DataFrame(self.return_std, index=self.date_list)
        temporary_impact_df = pd.DataFrame(self.temporary_impact, index=self.date_list)
        vwap400_df.to_csv("vwap400.csv")
        vwap330_df.to_csv("vwap330.csv")
        terminal_price_df.to_csv("terminal_price.csv")
        arrival_price_df.to_csv("arrival_price.csv")
        market_imbalance_df.to_csv("market_imbalance.csv")
        total_volume_df.to_csv("total_volume.csv")
        return_std_df.to_csv("return_std.csv")
        temporary_impact_df.to_csv("temporary_impact.csv")


if __name__ == '__main__':
    analysis = TAQAnalysis(QUOTE_DIR, TRADE_DIR, tickers, factor_df, split_df)
    analysis.run()
    analysis.save_results()
