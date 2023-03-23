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
            'Volume': self._s
        })
        df['Date'] = date + df['Date']
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H:%M:%S.%f')
        return df
