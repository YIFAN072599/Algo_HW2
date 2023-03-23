import gzip
import os.path
import pickle
import struct

import pandas as pd

from CollectTicker import collect_ticker
from Utils import milliseconds_to_time

WORK_DIR = os.path.dirname(__file__)
DATA_DIR = '/Users/chenzhao/Data/taq data'
QUOTE_DIR = os.path.join(DATA_DIR, 'quotes')
ADJ_QUOTE_DIR = os.path.join(WORK_DIR, 'data', 'quote')


class TAQQuotesReader(object):
    """
    This reader reads an entire compressed binary TAQ quotes file into memory,
    uncompresses it, and gives its clients access to the contents of the file
    via a set of get methods.
    """

    def __init__(self, filePathName):
        """
        Do all the heavy lifting here and give users getters for the
        results.
        """
        self._filePathName = filePathName
        with gzip.open(self._filePathName, 'rb') as f:
            file_content = f.read()
            self._header = struct.unpack_from(">2i", file_content[0:8])

            # millis from midnight
            endI = 8 + (4 * self._header[1])
            self._ts = struct.unpack_from((">%di" % self._header[1]), file_content[8:endI])
            startI = endI

            # bid size
            endI = endI + (4 * self._header[1])
            self._bs = struct.unpack_from((">%di" % self._header[1]), file_content[startI:endI])
            startI = endI

            # bid price
            endI = endI + (4 * self._header[1])
            self._bp = struct.unpack_from((">%df" % self._header[1]), file_content[startI:endI])
            startI = endI

            # ask size
            endI = endI + (4 * self._header[1])
            self._as = struct.unpack_from((">%di" % self._header[1]), file_content[startI:endI])
            startI = endI

            # ask price
            endI = endI + (4 * self._header[1])
            self._ap = struct.unpack_from((">%df" % self._header[1]), file_content[startI:endI])

    def getN(self):
        return self._header[1]

    def getSecsFromEpocToMidn(self):
        return self._header[0]

    def getMillisFromMidn(self, index):
        return self._ts[index]

    def getAskSize(self, index):
        return self._as[index]

    def getAskPrice(self, index):
        return self._ap[index]

    def getBidSize(self, index):
        return self._bs[index]

    def getBidPrice(self, index):
        return self._bp[index]

    def get_df(self, date):
        df = pd.DataFrame({
            'Date': [milliseconds_to_time(t) for t in self._ts],
            'AskPrice': self._ap,
            'AskSize': self._as,
            'BidPrice': self._bp,
            'BidSize': self._bs
        })
        df['Date'] = date + df['Date']
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H:%M:%S')
        return df


class QuotesReader(object):
    def __init__(self, dirPath, adjPath):
        self._dirPath = dirPath
        # use collect_ticker() to get all tickers
        #self._tickers = collect_ticker()
        self._tickers = ['MS', 'A']
        self._data = {}
        self._adjPath = adjPath

        for root, dirs, files in os.walk(self._dirPath):
            for dir in dirs:
                for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir)):
                    for subfile in subfiles:
                        ticker = subfile.split('_quote')[0]
                        if ticker not in self._tickers:
                            continue
                        filePathName = os.path.join(subroot, subfile)
                        reader = TAQQuotesReader(filePathName)
                        df = reader.get_df(dir)
                        self._data[ticker] = pd.concat([self._data.get(ticker, pd.DataFrame()), df], ignore_index=True)

    def save_tickers(self):
        if not os.path.exists(self._adjPath):
            os.makedirs(self._adjPath)
        for ticker, data in self._data.items():
            path = os.path.join(self._adjPath, f"{ticker}.csv")
            data.set_index('Date', drop=True, inplace=True)
            data.sort_index(inplace=True)
            data[['Adj_AskPrice', 'Adj_AskSize', 'Adj_BidPrice', 'Adj_BidSize']] = data[['AskPrice', 'AskSize', 'BidPrice', 'BidSize']]
            data['Mid_Price'] = (data['Adj_AskPrice'] + data['Adj_BidPrice'])/2
            data.to_csv(path)


if __name__ == '__main__':
    reader = QuotesReader(QUOTE_DIR, ADJ_QUOTE_DIR)
    reader.save_tickers()
