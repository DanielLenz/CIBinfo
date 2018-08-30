from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import pandas as pd
import numpy as np

from .. import this_project as P

__all__ = [
    'PlanckPR2',
]

class DifferenceSpectrum:
    """Base class for noise power spectra, based on various data splits.
    We use this primarily for the Planck data, where we have different splits
    by ring half/mission half/half survey.
    """
    _ell = None
    _Nl = None
    _raw_table = None
    _split_type = None
    _freq = None
    _unit = None

    def __init__(self, freq, split_type='ring', unit='Jy^2/sr'):
        self.freq = freq
        self.split_type = split_type
        self.unit = unit 

    @property
    def ell(self):
        raise NotImplementedError('Only in inherited class')

    @property
    def Nl(self, ):
        raise NotImplementedError('Only in inherited class')


class PlanckPR2(DifferenceSpectrum):
    """ Angular power spectra for the Planck PR2 half-difference maps.
    """
    def __init__(self, freq, split_type='ring', unit='Jy^2/sr'):
        super().__init__(
            freq=freq,
            split_type=split_type,
            unit=unit)

    @property
    def ell(self):
        if self._ell is None:
            self._ell = self.raw_table['ell'].values
        return self._ell

    @property
    def Nl(self, ):
        if self._Nl is None:
            # The native unit is Jy^2/sr
            self._Nl = self.raw_table[self.freq].values

            if self.unit == 'MJy^2/sr':
                self._Nl /= 1.e12
            if self.unit == 'K^2.sr':
                self._Nl *= (P.Jy2K[self.freq])**2
            if self.unit == 'uK^2.sr':
                self._Nl *= (P.Jy2K[self.freq])**2
                self._Nl *= 1.e12  # from K^2.sr to uK^2.sr

        return self._Nl

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        if not isinstance(value, str):
            raise TypeError('unit must be str')

        allowed = {'Jy^2/sr', 'MJy^2/sr', 'K^2.sr', 'uK^2.sr'}
        if value not in allowed:
            raise ValueError(f'unit must be in {allowed}')

        self._unit = value

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, value):
        if not (isinstance(value, str) or isinstance(value, int)):
            raise TypeError('freq must be str or int')

        value = str(value)

        # TODO go back and also add the 143 GHz data
        allowed = {'100', '217', '353', '545', '857'}

        if value not in allowed:
            raise ValueError(f'freq must be in {allowed}')

        self._freq = value

    @property
    def split_type(self):
        return self._split_type
    
    @split_type.setter
    def split_type(self, value):
        if not isinstance(value, str):
            raise TypeError('split_type must be str')
        
        value = value.lower()

        # TODO also add mission and survey splits
        allowed = {'ring', }

        if value not in allowed:
            raise ValueError(f'split_type must be in {allowed}')
        
        self._split_type = value

    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = pd.read_csv(
                    os.path.join(
                        P.PACKAGE_DIR,
                        'resources/noise/PlanckPR2_ring_differences.csv'),
                    comment='#')

        return self._raw_table
    
    @raw_table.setter
    def raw_table(self, value):
        raise RuntimeError('raw_table cannot be changed')


def main():
    pass

if __name__ == '__main__':
    main()