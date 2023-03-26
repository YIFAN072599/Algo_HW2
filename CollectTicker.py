import pandas as pd

sp_path = 'data/s&p500.xlsx'


def collect_ticker(path=sp_path):
    df = pd.read_excel(path, sheet_name="WRDS", engine="openpyxl")
    df_ticker = df["Trading Symbol"].unique()
    return df_ticker


if __name__ == '__main__':
    print(collect_ticker())
