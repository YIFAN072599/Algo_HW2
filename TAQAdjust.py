import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

WORK_DIR = os.path.dirname(__file__)
ADJ_QUOTE_DIR = os.path.join(WORK_DIR, 'data', 'quote')
ADJ_TRADE_DIR = os.path.join(WORK_DIR, 'data', 'trade')
SP_PATH = os.path.join(WORK_DIR, 'data', 's&p500.xlsx')
FIG_DIR = os.path.join(WORK_DIR, 'figure', 'adjust')


def get_factor(path=SP_PATH):
    df = pd.read_excel(path, usecols=["Names Date", "Trading Symbol", "Cumulative Factor to Adjust Prices",
                                      "Cumulative Factor to Adjust Shares/Vol"])
    df = df.groupby(["Names Date", "Trading Symbol", "Cumulative Factor to Adjust Prices",
                     "Cumulative Factor to Adjust Shares/Vol"]).size().reset_index(name="Frequency")
    df = df.rename(columns={"Names Date": "Date", "Trading Symbol": "Ticker"})
    df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d')
    return df


def get_adjust_date(path=SP_PATH):
    convert_date = lambda date_string: datetime.strptime(str(date_string), "%Y%m%d.%f").strftime("%Y-%m-%d")

    df_sp = pd.read_excel(path)
    df = df_sp.groupby(["Names Date", "Trading Symbol", "Cumulative Factor to Adjust Prices",
                        "Cumulative Factor to Adjust Shares/Vol"]).size().reset_index(name="Frequence")
    df_2 = df_sp.groupby(["Trading Symbol", "Cumulative Factor to Adjust Prices",
                          "Cumulative Factor to Adjust Shares/Vol"]).size().reset_index(name="Frequence")
    df_2 = df_2.groupby(["Trading Symbol"]).count()
    ticker_list = list(df_2.loc[df_2.Frequence > 1].index)
    df_ticker = df.loc[df["Trading Symbol"].isin(ticker_list)]
    df_ticker.reset_index(inplace=True, drop=True)
    date = []
    ticker = []
    for j in ticker_list:
        df_handle = df_ticker[df_ticker["Trading Symbol"] == j]
        df_handle.reset_index(inplace=True, drop=True)
        for i in range(1, len(df_handle)):
            if df_handle.iloc[i]["Cumulative Factor to Adjust Prices"] != df_handle.iloc[i - 1][
                "Cumulative Factor to Adjust Prices"] or df_handle.iloc[i]["Cumulative Factor to Adjust Shares/Vol"] != \
                    df_handle.iloc[i - 1]["Cumulative Factor to Adjust Shares/Vol"]:
                date.append(df_handle.loc[i, "Names Date"])
                ticker.append(df_handle.loc[i, "Trading Symbol"])
    split_df = {"ticker": ticker, "splitting_date": date}
    split_df = pd.DataFrame(split_df)
    # apply the conversion function to the date column
    split_df['splitting_date'] = split_df['splitting_date'].apply(convert_date)
    split_df.set_index('ticker', inplace=True)
    return split_df


class TAQAdjust:
    def __init__(self, quote_path, trade_path, sp_path):
        self._quote_path = quote_path
        self._trade_path = trade_path
        self._sp_path = sp_path
        self._factor_df = get_factor(sp_path)
        self._split_df = get_adjust_date(sp_path)
        workdir = os.listdir(quote_path)
        if '.DS_Store' in workdir:
            workdir.remove('.DS_Store')
        self._ticker = [x[:-4] for x in workdir]

    def adjust(self):
        for ticker in self._split_df.index:
            if ticker not in self._ticker:
                continue
            factor_df = self._factor_df[self._factor_df['Ticker'] == ticker].set_index('Date')
            date = self._split_df.loc[ticker, 'splitting_date']
            prev_index = factor_df.index.get_loc(date) - 1
            p_factor = factor_df.loc[date, 'Cumulative Factor to Adjust Prices'] / factor_df.iloc[prev_index, 1]
            v_factor = factor_df.loc[date, 'Cumulative Factor to Adjust Shares/Vol'] / factor_df.iloc[prev_index, 2]
            quote = pd.read_csv(os.path.join(self._quote_path, ticker + '.csv'), parse_dates=['Date'])
            trade = pd.read_csv(os.path.join(self._trade_path, ticker + '.csv'), parse_dates=['Date'])

            quote_idx = quote['Date'] < date
            trade_idx = trade['Date'] < date

            quote.loc[quote_idx, 'Adj_AskPrice'] = quote.loc[quote_idx, 'AskPrice'] * p_factor
            quote.loc[quote_idx, 'Adj_BidPrice'] = quote.loc[quote_idx, 'BidPrice'] * p_factor
            quote.loc[quote_idx, 'Adj_AskSize'] = quote.loc[quote_idx, 'AskSize'] * v_factor
            quote.loc[quote_idx, 'Adj_BidSize'] = quote.loc[quote_idx, 'BidSize'] * v_factor

            trade.loc[trade_idx, 'Adj_Price'] = trade.loc[trade_idx, 'Price'] * p_factor
            trade.loc[trade_idx, 'Adj_Size'] = trade.loc[trade_idx, 'Size'] * v_factor

            quote['Mid_Price'] = (quote['Adj_AskPrice'] + quote['Adj_BidPrice']) / 2

            quote.to_csv(os.path.join(self._quote_path, ticker + '.csv'), index=False)
            trade.to_csv(os.path.join(self._trade_path, ticker + '.csv'), index=False)

    def plot(self):
        if not os.path.exists(FIG_DIR):
            os.mkdir(FIG_DIR)
        plt.figure(figsize=(16, 9))
        for ticker in self._ticker:
            quote = pd.read_csv(os.path.join(self._quote_path, ticker + '.csv'), parse_dates=['Date'])
            plt.plot(quote['Date'], quote['Adj_AskPrice'], label=ticker + ' Adj_AskPrice')
            plt.plot(quote['Date'], quote['Adj_BidPrice'], label=ticker + ' Adj_BidPrice')
            plt.legend()
            plt.title(f'{ticker} Quote Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.savefig(os.path.join(FIG_DIR, f'{ticker}_quote_adj_prices.png'))
            plt.clf()

        # Plot trade's Adj_Price in a separate figure
        plt.figure(figsize=(16, 9))
        for ticker in self._ticker:
            trade = pd.read_csv(os.path.join(self._trade_path, ticker + '.csv'), parse_dates=['Date'])
            plt.plot(trade['Date'], trade['Adj_Price'], label=ticker + ' Adj_Price')
            plt.legend()
            plt.title(f'{ticker} Trade Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.savefig(os.path.join(FIG_DIR, f'{ticker}_trade_adj_prices.png'))
            plt.clf()


if __name__ == '__main__':
    adjust = TAQAdjust(ADJ_QUOTE_DIR, ADJ_TRADE_DIR, SP_PATH)
    #adjust.adjust()
    adjust.plot()
