import gzip
import os.path
import struct

import pandas as pd
import numpy as np

from CollectTicker import collect_ticker
from Utils import milliseconds_to_time

WORK_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORK_DIR, 'data')
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
            self._ts = np.asarray(struct.unpack_from((">%di" % self._header[1]), file_content[8:endI]))
            startI = endI

            # bid size
            endI = endI + (4 * self._header[1])
            self._bs = np.asarray(struct.unpack_from((">%di" % self._header[1]), file_content[startI:endI]))
            startI = endI

            # bid price
            endI = endI + (4 * self._header[1])
            self._bp = np.asarray(struct.unpack_from((">%df" % self._header[1]), file_content[startI:endI]))
            startI = endI

            # ask size
            endI = endI + (4 * self._header[1])
            self._as = np.asarray(struct.unpack_from((">%di" % self._header[1]), file_content[startI:endI]))
            startI = endI

            # ask price
            endI = endI + (4 * self._header[1])
            self._ap = np.asarray(struct.unpack_from((">%df" % self._header[1]), file_content[startI:endI]))

    def get_df(self, date, ticker):
        ts = np.vectorize(milliseconds_to_time)(self._ts)
        ts = ts.astype(date.dtype)
        # Concatenate the arrays
        combined = np.core.defchararray.add(date, ts)
        dates = pd.to_datetime(combined, format='%Y%m%d%H:%M:%S.%f')
        tickers = np.full_like(self._ts, ticker, dtype=object)

        df = pd.DataFrame({
            'Date': dates,
            'Ticker': tickers,
            'AskPrice': self._ap,
            'AskSize': self._as,
            'BidPrice': self._bp,
            'BidSize': self._bs
        })

        return df


