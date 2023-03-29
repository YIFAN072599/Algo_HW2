import gzip
import os
import struct

import pandas as pd
import numpy as np

from Utils import milliseconds_to_time

WORK_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORK_DIR, 'data')
TRADE_DIR = os.path.join(DATA_DIR, 'trades')
ADJ_TRADE_DIR = os.path.join(WORK_DIR, 'data', 'trade')


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
            self._ts = np.asarray(struct.unpack_from((">%di" % self._header[1]), file_content[8:endI]))
            startI = endI
            endI = endI + (4 * self._header[1])
            self._s = np.asarray(struct.unpack_from((">%di" % self._header[1]), file_content[startI:endI]))
            startI = endI
            endI = endI + (4 * self._header[1])
            self._p = np.asarray(struct.unpack_from((">%df" % self._header[1]), file_content[startI:endI]))

    def get_df(self, date):
        df = pd.DataFrame({
            'Date': [milliseconds_to_time(t) for t in self._ts],
            'Price': self._p,
            'Volume': self._s
        })
        df['Date'] = date + df['Date']
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H:%M:%S.%f')
        return df
