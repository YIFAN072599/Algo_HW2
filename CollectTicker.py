import os

import pandas as pd

sp_path = 'data/s&p500.xlsx'
src_dir = "C:\\Users\\Admin\\Downloads\\Algo HW2\\Algo HW2\\quotes\\quotes"

def collect_ticker(path=sp_path):
    df = pd.read_excel(path, sheet_name="WRDS", engine="openpyxl")
    df_ticker = df["Trading Symbol"].unique()
    return df_ticker

def collect_date(path=src_dir):
    return os.listdir(src_dir)

if __name__ == '__main__':
    print(len(collect_date()))
