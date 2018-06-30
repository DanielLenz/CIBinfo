from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pandas as pd
import os

from .. import this_project as P


__all__ = [
    'Planck13Data',
    'Planck13Model',
    'Maniyar18Model',
    'GNILCxPlanckPR2', ]


class CIBxPhi():

    _l = None  # multipole

    _Cl = None  # angular power
    _dCl = None  # uncertainty on the angular power

    _l3Cl = None  # l**3 * Cl
    _dl3Cl = None  # uncertainty on l**3 * Cl

    _raw_table = None  # raw table, taken from publications, emails, etc

    def __init__(self, freq, unit='Jy'):
        if unit not in ['Jy', 'MJy', 'uK.sr']:
            raise ValueError('Unit must be either "Jy", "MJy", or "uK.sr"')
        self.unit = unit

        self.freq = freq

    # Properties
    ############
    @property
    def freqstr(self):
        self._freqstr = str(self.freq)
        return self._freqstr

    @property
    def l(self):
        if self._l is None:
            self._l = self.raw_table[:, 0]
        return self._l

    @property
    def Jy2K(self):
        self._Jy2K = P.Jy2K
        return self._Jy2K

    @property
    def K2Jy(self):
        self._K2Jy = P.K2Jy
        return self._K2Jy


class Planck13Data(CIBxPhi):
    def __init__(self, freq, unit='uK.sr'):
        super(Planck13Data, self).__init__(freq, unit=unit)

    # Properties
    ############
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = np.genfromtxt(os.path.join(
                P.PACKAGE_DIR,
                'resources/cibxphi/Planck13_data_{}.dat'.format(self.freq)),
                usecols=[0, 1, 2])
        return self._raw_table

    @property
    def l3Cl(self):
        if self._l3Cl is None:
            # native unit is uK*sr
            self._l3Cl = self.raw_table[:, 1].copy()

            if self.unit == 'Jy':
                self._l3Cl *= self.K2Jy[self.freqstr] / 1.e6
            if self.unit == 'MJy':
                self._l3Cl *= self.K2Jy[self.freqstr] / 1.e12

        return self._l3Cl

    @property
    def Cl(self):
        if self._Cl is None:
            self._Cl = self.l3Cl / self.l**3

        return self._Cl

    @property
    def dl3Cl(self):
        if self._dl3Cl is None:
            # native unit is uK*sr
            self._dl3Cl = self.raw_table[:, 2].copy()

            if self.unit == 'Jy':
                self._dl3Cl *= self.K2Jy[self.freqstr] / 1.e6
            if self.unit == 'MJy':
                self._dl3Cl *= self.K2Jy[self.freqstr] / 1.e12
        return self._dl3Cl

    @property
    def dCl(self):
        if self._dCl is None:
            self._dCl = self.dl3Cl / self.l**3

        return self._dCl


class Planck13Model(CIBxPhi):
    def __init__(self, freq, unit='Jy'):
        super(Planck13Model, self).__init__(freq, unit=unit)

    # Properties
    ############
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = np.genfromtxt(os.path.join(
                P.PACKAGE_DIR,
                'resources/cibxphi/Planck13_model_{}.txt'.format(self.freq)),
                usecols=[0, 1])
        return self._raw_table

    @property
    def l3Cl(self):
        if self._l3Cl is None:
            # native unit is Jy
            self._l3Cl = self.raw_table[:, 1].copy()

            if self.unit == 'uK.sr':
                self._l3Cl *= self.Jy2K[self.freqstr] * 1.e6
            if self.unit == 'MJy':
                self._l3Cl /= 1.e6

        return self._l3Cl

    @property
    def Cl(self):
        if self._Cl is None:
            self._Cl = self.l3Cl / self.l**3
        return self._Cl


class Maniyar18Model(CIBxPhi):
    def __init__(self, freq, unit='Jy'):
        super(Maniyar18Model, self).__init__(freq, unit=unit)

    # Properties
    ############
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = np.loadtxt(
                os.path.join(
                    P.PACKAGE_DIR,
                    'resources/cibxphi/Maniyar18_model.dat'))
        return self._raw_table

    @property
    def l3Cl(self):
        if self._l3Cl is None:
            # native unit is Jy
            self._l3Cl = self.raw_table[:, self._freq2col(self.freqstr)].copy()

            if self.unit == 'uK.sr':
                self._l3Cl *= self.Jy2K[self.freqstr] * 1.e6
            if self.unit == 'MJy':
                self._l3Cl /= 1.e6

        return self._l3Cl

    @property
    def Cl(self):
        if self._Cl is None:
            self._Cl = self.l3Cl / self.l**3
        return self._Cl

    # Methods
    #########
    def _freq2col(self, freqstr):
        # ell, Phix100, Phix143, Phix217, Phix353, Phix545, Phix857
        mapping = {
            '100': 1,
            '143': 2,
            '217': 3,
            '353': 4,
            '545': 5,
            '857': 6,
        }

        return mapping[self.freqstr]


class GNILCxPlanckPR2(CIBxPhi):
    def __init__(self, freq, unit='Jy'):
        super().__init__(freq, unit=unit)

    # Properties
    ############
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = pd.read_csv(
                os.path.join(
                    P.PACKAGE_DIR,
                    f'resources/cibxphi/df_gnilcxphi_binned_{self.freq}.csv'),
                comment='#')

        return self._raw_table

    @property
    def l(self):
        if self._l is None:
            self._l = self.raw_table['b'].values
        return self._l

    @property
    def dl(self):
        if self._dl is None:
            self._dl = self.raw_table['db'].values
        return self._dl

    @property
    def Cl(self):
        if self._Cl is None:
            self._Cl = self.l3Cl / self.l**3

        return self._Cl

    @property
    def dCl(self):
        if self._dCl is None:
            self._dCl = self.l3dCl / self.l**3
        return self._dCl

    @property
    def l3Cl(self):
        if self._l3Cl is None:

            # Native unit is Jy
            self._l3Cl = self.raw_table['b3Cb'].values

            if self.unit == 'uK.sr':
                self._l3Cl *= self.Jy2K[self.freqstr] * 1.e6
            if self.unit == 'MJy':
                self._l3Cl /= 1.e6

        return self._l3Cl

    @property
    def dl3Cl(self):
        if self._dl3Cl is None:
            # Native unit is Jy
            self._dl3Cl = self.raw_table['b3dCb'].values

            if self.unit == 'uK.sr':
                self._dl3Cl *= self.Jy2K[self.freqstr] * 1.e6
            if self.unit == 'MJy':
                self._dl3Cl /= 1.e6

        return self._dl3Cl
