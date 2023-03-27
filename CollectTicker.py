import os

import pandas as pd

sp_path = 'data/s&p500.xlsx'
src_dir = "/Users/chenzhao/Data/taq data/quotes"


def collect_ticker(path=sp_path):
    df = pd.read_excel(path, sheet_name="WRDS", engine="openpyxl")
    df_ticker = list(df["Trading Symbol"].unique())
    df_ticker.remove('ESRX')
    return df_ticker


def collect_date(path=src_dir):
    return os.listdir(path)


if __name__ == '__main__':
    print('ESRX' in collect_ticker())
