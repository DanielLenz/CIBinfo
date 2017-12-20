from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os

from .. import this_project as P


class AutoPowerspectrum():

    _l = None  # multipole
    _Cl = None  # angular power
    _dCl = None  # uncertainty on the angular power
    _Dl = None  # l(l+1)/2pi * Cl
    _l3Cl = None  # l**3 * Cl
    _S = None  # shot noise level
    _dS = None  # uncertainty on the shot noise

    _raw_table = None  # raw table, taken from publications, emails, etc

    def __init__(self, freq1, freq2=None, unit='Jy^2/sr'):
        if unit not in ['Jy^2/sr', 'MJy^2/sr', 'K^2.sr']:
            raise ValueError('Unit must be either "Jy^2/sr" or "K^2.sr"')
        self.unit = unit

        self.freq1 = freq1
        if freq2 is None:
            self.freq2 = freq1
        else:
            self.freq2 = freq2

    # Properties
    ###########################################################################
    @property
    def freqstr(self):
        self._freqstr = 'x'.join((
            str(max([self.freq1, self.freq2])),
            str(min([self.freq1, self.freq2]))
            ))
        return self._freqstr

    @property
    def l(self):
        if self._l is None:
            self._l = self.raw_table[:, 0]
        return self._l

    @property
    def Cl(self):
        return None

    @property
    def dCl(self):
        return None

    @property
    def Dl(self):
        if self._Dl is None:
            self._Dl = l*(l+1.)/2./np.pi * self.Cl
        return self._Dl

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

    # methods
    ###########################################################################


class Planck2014(AutoPowerspectrum):
    def __init__(self, freq1, freq2=None, unit='Jy^2/sr'):
        super(Planck2014, self).__init__(freq1, freq2=freq2, unit=unit)

        self.Cl_contains_SN = False

    # Properties
    ###########################################################################
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = np.loadtxt(os.path.join(
                P.PACKAGE_DIR,
                'resources/planck_2014_XXX/all_best_fit.txt'
                ))
        return self._raw_table

    @property
    def Cl(self):
        if self._Cl is None:
            # native unit is Jy^2/sr
            self._Cl = self.raw_table[:, self._freq2col(self.freqstr)].copy()

            if self.unit == 'K^2*sr':
                self._Cl *= (
                    self.Jy2K[str(self.freq1)] *
                    self.Jy2K[str(self.freq2)])

            if self.unit == 'MJy^2/sr':
                self._Cl /= 1.e12

        return self._Cl

    @property
    def S(self):
        """
        Shot noise
        Units are Jy^2/sr
        """
        if self._S is None:
            self._S = {
                '857x857': 5364,
                '545x545': 1690,
                '353x353': 262,
                '217x217': 21,
                '3000x3000': 9585,
                '857x545': 2702,
                '857x353': 953,
                '857x217': 181,
                '545x353': 626,
                '545x217': 121,
                '353x217': 54,
                '3000x857': 4158,
                '3000x545': 1449,
                '3000x353': 411,
                '3000x217': 95,
            }

            if self.unit == 'K^2*sr':
                self._S *= (
                    self.Jy2K[str(self.freq1)] *
                    self.Jy2K[str(self.freq2)])

            if self.unit == 'MJy^2/sr':
                self._S /= 1.e12

        return self._S[self.freqstr]

    @property
    def dS(self):
        """
        Shot noise
        """
        if self._dS is None:
            self._dS = {
                '857x857': 343,
                '545x545': 45,
                '353x353': 8,
                '217x217': 2,
                '3000x3000': 1090,
                '857x545': 124,
                '857x353': 54,
                '857x217': 6,
                '545x353': 19,
                '545x217':  6,
                '353x217':  3,
                '3000x857': 443,
                '3000x545': 176,
                '3000x353': 48,
                '3000x217': 11,
            }
            if self.unit == 'K^2*sr':
                self._dS *= (
                    self.Jy2K[str(self.freq1)] *
                    self.Jy2K[str(self.freq2)])

            if self.unit == 'MJy^2/sr':
                self._dS /= 1.e12

        return self._dS[self.freqstr]

    # methods
    ###########################################################################
    def _freq2col(self, freqstr):
        mapping = {
            '857x857': 1,
            '545x545': 2,
            '353x353': 3,
            '217x217': 4,
            '3000x3000': 5,
            '857x545': 6,
            '857x353': 7,
            '857x217': 8,
            '545x353': 9,
            '545x217': 10,
            '353x217': 11,
            '3000x857': 12,
            '3000x545': 13,
            '3000x353': 14,
            '3000x217': 15,
            '100x100': 16,
            '857x100': 17,
            '545x100': 18,
            '353x100': 19,
            '217x100': 20,
            '143x143': 21,
            '857x143': 22,
            '545x143': 23,
            '353x143': 24,
            '217x143': 25,
            '143x100': 26
        }

        return mapping[self.freqstr]


class PaoloModel(AutoPowerspectrum):
    def __init__(self, freq1, freq2=None, unit='Jy^2/sr'):
        super(PaoloModel, self).__init__(freq1, freq2=freq2, unit=unit)

        self.Cl_contains_SN = True

    # Properties
    ###########################################################################
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = np.loadtxt(os.path.join(
                P.PACKAGE_DIR,
                'resources/paolo_models/cib_all_spectra_l3000.txt'
                ))
        return self._raw_table

    @property
    def Cl(self):
        if self._Cl is None:
            # native unit is be Jy^2/sr
            self._Cl = self.raw_table[:, self._freq2col(self.freqstr)].copy()

            # Possibly convert the units
            if self.unit == 'K^2.sr':
                self._Cl *= (
                    self.Jy2K[str(self.freq1)] *
                    self.Jy2K[str(self.freq2)]
                    )

            if self.unit == 'MJy^2/sr':
                self._Cl /= 1.e12

        return self._Cl

    # methods
    ###########################################################################
    def _freq2col(self, freqstr):
        # l, 353x353, 353x545, 353x857, 545x545, 545x857, 857x857
        mapping = {
            '353x353': 1,
            '545x353': 2,
            '857x353': 3,
            '545x545': 4,
            '857x545': 5,
            '857x857': 6,
        }

        return mapping[self.freqstr]


class AbhiModel(AutoPowerspectrum):
    def __init__(self, freq1, freq2=None, unit='Jy^2/sr'):
        super(AbhiModel, self).__init__(freq1, freq2=freq2, unit=unit)

        self.Cl_contains_SN = True
        self.Cl_contains_1halo = True

    # Properties
    ###########################################################################
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = np.loadtxt(os.path.join(
                P.PACKAGE_DIR,
                'resources/cibxcib/abhimodel_cibxcib.dat'
                ))
        return self._raw_table

    @property
    def Cl(self):
        if self._Cl is None:
            # native unit is be Jy^2/sr
            self._Cl = self.raw_table[:, self._freq2col(self.freqstr)].copy()

            # Possibly convert the units
            if self.unit == 'K^2.sr':
                self._Cl *= (
                    self.Jy2K[str(self.freq1)] *
                    self.Jy2K[str(self.freq2)]
                    )

            if self.unit == 'MJy^2/sr':
                self._Cl /= 1.e12

        return self._Cl

    # Methods
    ###########################################################################
    def _freq2col(self, freqstr):
        # 217x217, 353x353, 545x545, 857x857, 3000x3000
        mapping = {
            '217x217': 1,
            '353x353': 2,
            '545x545': 3,
            '857x857': 4,
            '3000x3000': 5,
        }

        return mapping[self.freqstr]
