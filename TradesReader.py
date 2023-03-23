import gzip
import os
import struct

import pandas as pd

from Utils import milliseconds_to_time

WORK_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORK_DIR, 'data')
TRADE_DIR = os.path.join(DATA_DIR, 'trades')
ADJ_TRADE_DIR = os.path.join(WORK_DIR, 'data', 'trade')


def convert_date(df, col_name):
    # create a copy of the input DataFrame
    df_new = df.copy()

    # convert the specified column to datetime format
    df_new[col_name] = pd.to_datetime(df_new[col_name], format='%Y%m%d')

    # format the datetime column as a string in 'YYYY-MM-DD' format
    df_new[col_name] = df_new[col_name].dt.strftime('%Y-%m-%d')

    return df_new


class TAQTradesReader(object):
    """
    This reader reads an entire compressed binary TAQ trades file into memory,
    uncompresses it, and gives its clients access to the contents of the file
    via a set of get methods.
    """

    def __init__(self, filePathName):
        """
        Do all the heavy lifting here and give users getters for the results.
        """
        self.filePathName = filePathName
        with gzip.open(filePathName, 'rb') as f:
            file_content = f.read()
            self._header = struct.unpack_from(">2i", file_content[0:8])
            endI = 8 + (4 * self._header[1])
            self._ts = struct.unpack_from((">%di" % self._header[1]), file_content[8:endI])
            startI = endI
            endI = endI + (4 * self._header[1])
            self._s = struct.unpack_from((">%di" % self._header[1]), file_content[startI:endI])
            startI = endI
            endI = endI + (4 * self._header[1])
            self._p = struct.unpack_from((">%df" % self._header[1]), file_content[startI:endI])

    def getN(self):
        return self._header[1]

    def getSecsFromEpocToMidn(self):
        return self._header[0]

    def getPrice(self, index):
        return self._p[index]

    def getMillisFromMidn(self, index):
        return self._ts[index]

    def getTimestamp(self, index):
        return self.getMillisFromMidn(index)  # Compatibility

    def getSize(self, index):
        return self._s[index]

    def rewrite(self, filePathName, tickerId):
        s = struct.Struct(">QHIf")
        out = gzip.open(filePathName, "wb")
        baseTS = self.getSecsFromEpocToMidn() * 1000
        for i in range(self.getN()):
            ts = baseTS + self.getMillisFromMidn(i)
            out.write(s.pack(ts, tickerId, self.getSize(i), self.getPrice(i)))
        out.close()

    def get_df(self, date):
        df = pd.DataFrame({
            'Date': [milliseconds_to_time(t) for t in self._ts],
            'Price': self._p,
            'Size': self._s
        })
        df['Date'] = date + df['Date']
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H:%M:%S')
        return df


class TradesReader(object):
    def __init__(self, dirPath, adjPath):
        self._adjPath = adjPath
        self._dirPath = dirPath
        # use collect_ticker() to get all tickers
        # self._tickers = collect_ticker()
        self._tickers = ['MS', 'A']
        self._data = {}

        for root, dirs, files in os.walk(self._dirPath):
            for dir in dirs:
                for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir)):
                    for subfile in subfiles:
                        ticker = subfile.split('_trade')[0]
                        if ticker not in self._tickers:
                            continue
                        filePathName = os.path.join(subroot, subfile)
                        reader = TAQTradesReader(filePathName)
                        df = reader.get_df(dir)
                        self._data[ticker] = pd.concat([self._data.get(ticker, pd.DataFrame()), df], ignore_index=True)

    def save_tickers(self):
        if not os.path.exists(self._adjPath):
            os.makedirs(self._adjPath)
        for ticker, data in self._data.items():
            path = os.path.join(self._adjPath, f"{ticker}.csv")
            data.set_index('Date', drop=True, inplace=True)
            data.sort_index(inplace=True)
            data[['Adj_Price', 'Adj_Size']] = data[['Price', 'Size']]
            data.to_csv(path)


if __name__ == '__main__':
    reader = TradesReader(TRADE_DIR, ADJ_TRADE_DIR)
    reader.save_tickers()
