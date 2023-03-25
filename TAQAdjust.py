import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

WORK_DIR = os.path.dirname(__file__)
SP_PATH = os.path.join(WORK_DIR, 'data', 's&p500.xlsx')


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
    def __init__(self, df, sp_path=SP_PATH):
        self._df = df
        self._sp_path = sp_path
        self._factor_df = get_factor(sp_path)
        self._split_df = get_adjust_date(sp_path)
        self._date = df['Date']
        self._ticker = df['Ticker'].unique()[0]

    def adjust(self):
        if self._ticker in self._split_df.index:
            factor_df = self._factor_df[self._factor_df['Ticker'] == self._ticker].set_index('Date')
            date = self._split_df.loc[self._ticker, 'splitting_date']
            prev_index = factor_df.index.get_loc(date) - 1
            p_factor = factor_df.loc[date, 'Cumulative Factor to Adjust Prices'] / factor_df.iloc[prev_index, 1]
            v_factor = factor_df.loc[date, 'Cumulative Factor to Adjust Shares/Vol'] / factor_df.iloc[prev_index, 2]

            idx = self._date < date

            self._df.loc[idx, 'AskPrice'] = self._df.loc[idx, 'AskPrice'] * p_factor
            self._df.loc[idx, 'BidPrice'] = self._df.loc[idx, 'BidPrice'] * p_factor
            self._df.loc[idx, 'AskSize'] = self._df.loc[idx, 'AskSize'] * v_factor
            self._df.loc[idx, 'BidSize'] = self._df.loc[idx, 'BidSize'] * v_factor

            self._df.loc[idx, 'Adj_Price'] = self._df.loc[idx, 'Price'] * p_factor
            self._df.loc[idx, 'Adj_Size'] = self._df.loc[idx, 'Size'] * v_factor

    def get_df(self):
        return self._df
