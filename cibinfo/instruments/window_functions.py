from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import warnings

import numpy as np
import pandas as pd

from .. import this_project as P


__all__ = [
    'SMICA',
]


class WindowFunction:
    _raw_table = None

    _ell = None
    _Bl = None  # Beam window
    _Pl = None  # Pixel window
    _Wl = None  # Effective window function (often Bl x Pl)

    def __init__(self):
        ...

    @property
    def ell(self):
        if self._ell is None:
            self._ell = self.raw_table['ell'].values

        return self._ell

    @property
    def Bl(self):
        raise NotImplementedError('Not implemented in base class')

    @property
    def Wl(self):
        raise NotImplementedError('Not implemented in base class')

    @property
    def Pl(self):
        raise NotImplementedError('Not implemented in base class')


class SMICA(WindowFunction):
    def __init__(self):
        super().__init__()


    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = pd.read_csv(
                os.path.join(
                    P.PACKAGE_DIR,
                    'resources/instruments/Bl_smica.csv'),
                comment='#',)
        
        return self._raw_table

    @property
    def Pl(self):
        if self._Pl is None:
            warnings.warn(
                'The SMICA map is deconvolved from the pixel window, '
                'hence Pl is just unity.')
            
            self._Pl = np.ones_like(self.Bl, dtype=np.double)
        return self._Pl 

    @property
    def Bl(self):
        if self._Bl is None:
            self._Bl = self.raw_table['Bl'].values

        return self._Bl

    @property
    def Wl(self):
        if self._Wl is None:
            self._Wl = self.Pl * self.Bl

        return self._Wl


def main():
    pass

if __name__ == '__main__':
    main()