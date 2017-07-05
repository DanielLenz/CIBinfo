from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os

from .. import this_project as P


class CrossPowerspectrum():

    _l = None  # multipole

    _Cl = None  # angular power
    _dCl = None  # uncertainty on the angular power

    _l3Cl = None  # l**3 * Cl
    _dl3Cl = None  # uncertainty on l**3 * Cl

    _raw_table = None  # raw table, taken from publications, emails, etc

    def __init__(self, freq, unit='Jy'):
        if unit not in ['Jy', 'uK*sr']:
            raise ValueError('Unit must be either "Jy" or "K*sr"')
        self.unit = unit

        self.freq = freq

    # properties
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
    def l3Cl(self):
        if self._l3Cl is None:
            self._l3Cl = self.l**3 * self.Cl
        return self._l3Cl

    @property
    def Jy2K(self):
        self._Jy2K = P.Jy2K
        return self._Jy2K

    @property
    def K2Jy(self):
        self._K2Jy = P.K2Jy
        return self._K2Jy

#
# class Planck2013(CrossPowerspectrum):
#     def __init__(self, freq, unit='K*sr'):
#         super(Planck2013, self).__init__(freq, unit=unit)
#
#     # properties
#     ############
#     @property
#     def raw_table(self):
#         if self._raw_table is None:
#             self._raw_table = np.loadtxt(os.path.join(
#                 P.PACKAGE_DIR,
#                 'resources/lensing_HOD/lensing_{}.txt'.format(self.freq)
#                 ))
#         return self._raw_table
#
#     @property
#     def Cl(self):
#         if self._Cl is None:
#             # native unit is muK*sr
#             self._Cl = self.raw_table[:, 1].copy()
#             self._Cl /= (self.l+1.)**3  # from l^3*Cl to Cl
#             self._Cl /= 1.e6  # from muK*sr to K*sr
#
#             if self.unit == 'Jy':
#                 self._Cl *= self.K2Jy[self.freqstr]
#         return self._Cl


class Planck2013(CrossPowerspectrum):
    def __init__(self, freq, unit='uK*sr'):
        super(Planck2013, self).__init__(freq, unit=unit)

    # properties
    ###########################################################################
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = np.genfromtxt(os.path.join(
                P.PACKAGE_DIR,
                'resources/cibxphi/Cl_{}_olivier.dat'.format(self.freq)),
                usecols=[0, 1, 2])
        return self._raw_table

    @property
    def l3Cl(self):
        if self._l3Cl is None:
            # native unit is uK*sr
            self._l3Cl = self.raw_table[:, 1].copy()

            if self.unit == 'Jy':
                self._l3Cl *= self.K2Jy[self.freqstr] / 1.e6
        return self._l3Cl

    @property
    def dl3Cl(self):
        if self._dl3Cl is None:
            # native unit is uK*sr
            self._dl3Cl = self.raw_table[:, 2].copy()

            if self.unit == 'Jy':
                self._dl3Cl *= self.K2Jy[self.freqstr] / 1.e6
        return self._dl3Cl


class Model(CrossPowerspectrum):
    def __init__(self, freq, unit='Jy'):
        super(Model, self).__init__(freq, unit=unit)

    # properties
    ###########################################################################
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = np.genfromtxt(os.path.join(
                P.PACKAGE_DIR,
                'resources/cibxphi/cl_phixT_{}.txt'.format(self.freq)),
                usecols=[0, 1])
        return self._raw_table

    @property
    def l3Cl(self):
        if self._l3Cl is None:
            # native unit is uK*sr
            self._l3Cl = self.raw_table[:, 1].copy()

            if self.unit == 'uK*sr':
                self._l3Cl *= self.Jy2K[self.freqstr] * 1.e6
        return self._l3Cl
