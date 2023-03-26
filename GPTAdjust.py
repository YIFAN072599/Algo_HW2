import os

import pandas as pd

WORK_DIR = os.path.dirname(__file__)
SP_PATH = os.path.join(WORK_DIR, 'data', 's&p500.xlsx')


def prepare_adjustment_data(path=SP_PATH):
    factor_df = pd.read_excel(path, usecols=["Names Date", "Trading Symbol", "Cumulative Factor to Adjust Prices",
                                             "Cumulative Factor to Adjust Shares/Vol"])
    factor_df = factor_df.groupby(["Names Date", "Trading Symbol", "Cumulative Factor to Adjust Prices",
                                   "Cumulative Factor to Adjust Shares/Vol"]).size().reset_index(name="Frequency")
    factor_df["Names Date"] = pd.to_datetime(factor_df["Names Date"], format='%Y%m%d')
    factor_df = factor_df.rename(columns={"Names Date": "Date", "Trading Symbol": "Ticker"})

    tickers_with_multiple_factors = factor_df.groupby("Ticker").filter(lambda x: len(x) > 1)
    split_dates = []
    tickers = []

    for ticker, ticker_data in tickers_with_multiple_factors.groupby("Ticker"):
        ticker_data = ticker_data.reset_index(drop=True)
        for i in range(1, len(ticker_data)):
            if ticker_data.iloc[i]["Cumulative Factor to Adjust Prices"] != ticker_data.iloc[i - 1][
                "Cumulative Factor to Adjust Prices"] or ticker_data.iloc[i][
                "Cumulative Factor to Adjust Shares/Vol"] != \
                    ticker_data.iloc[i - 1]["Cumulative Factor to Adjust Shares/Vol"]:
                split_dates.append(ticker_data.loc[i, "Date"])
                tickers.append(ticker_data.loc[i, "Ticker"])

    split_df = pd.DataFrame({"ticker": tickers, "splitting_date": split_dates})
    split_df["splitting_date"] = split_df["splitting_date"].dt.strftime('%Y-%m-%d')
    split_df.set_index('ticker', inplace=True)

    return factor_df, split_df


def adjust_taq_data(df, factor_df, split_df):
    df['Date'] = pd.to_datetime(df['Date'])
    factor_df.dropna(inplace=True)
    merged_df = pd.merge_asof(df, factor_df, on='Date', by='Ticker', direction='backward')

    for ticker in split_df.index.unique():
        idx = (merged_df['Ticker'] == ticker) & (merged_df['Date'] < split_df.loc[ticker, 'splitting_date'])
        merged_df.loc[idx, ['AskPrice', 'BidPrice', 'Price']] *= merged_df.loc[
            idx, 'Cumulative Factor to Adjust Prices']
        merged_df.loc[idx, ['AskSize', 'BidSize', 'Volume']] *= merged_df.loc[
            idx, 'Cumulative Factor to Adjust Shares/Vol']

    merged_df.drop(columns=['Cumulative Factor to Adjust Prices', 'Cumulative Factor to Adjust Shares/Vol'],
                   inplace=True)
    return merged_df


factor_df, split_df = prepare_adjustment_data()

# Use adjust_taq_data() with the prepared factor_df and split_df
# adjusted_data = adjust_taq_data(raw_data, factor_df, split_df)
if __name__ == '__main__':
    print(factor_df)
