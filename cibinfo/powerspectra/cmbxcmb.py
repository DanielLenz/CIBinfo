from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import os

from .. import this_project as P


__all__ = ["Planck15Model", "Planck18Data"]


class CMBxCMB:

    _l = None  # multipole

    _Cl = None  # angular power
    _dCl = None  # uncertainty on the angular power

    _l2Cl = None  # l(l+1)/2pi * Cl
    _dl2Cl = None  # uncertainty on l(l+1)/2pi * Cl

    _raw_table = None  # raw table, taken from publications, emails, etc

    def __init__(self, freq=None, mode="TT", unit="uK^2.sr"):
        if unit not in ["Jy^2/sr", "MJy^2/sr", "uK^2.sr", "K^2.sr"]:
            raise ValueError(
                'Unit must be either "Jy^2/sr",' '"MJy^2/sr", "K^2.sr", or "uK^2.sr"'
            )
        self.unit = unit

        if (unit not in ["uK^2.sr", "K^2.sr"]) and (freq is None):
            raise ValueError("If unit is frequency-dependent, freq must be provided.")

        mode = mode.upper()
        if mode not in ["TT", "TE", "EE", "BB"]:
            raise ValueError("Mode must be in ['TT', 'TE', 'EE', 'BB']")
        self.mode = mode

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


class Planck15Model(CMBxCMB):
    def __init__(self, freq=None, mode="TT", unit="uK^2.sr"):
        super().__init__(freq=freq, mode=mode, unit=unit)

    # Properties
    ############
    @property
    def raw_table(self):
        if self._raw_table is None:
            _data = np.loadtxt(
                os.path.join(
                    P.PACKAGE_DIR,
                    "resources/cmbxcmb/COM_PowerSpect_CMB-base-plikHM-TT-lowTEB"
                    "-minimum-theory_R2.02.txt",
                )
            )

            self._raw_table = pd.DataFrame(
                data=_data, columns=["L", "TT", "TE", "EE", "BB", "PP"]
            )
        return self._raw_table

    @property
    def l(self):
        if self._l is None:
            self._l = self.raw_table.L.values
        return self._l

    @property
    def l_scaling(self):
        return self.l * (self.l + 1) / 2.0 / np.pi

    @property
    def l2Cl(self):
        """l(l+1)/2pi * Cl"""
        if self._l2Cl is None:
            # Native unit is uK^2/sr
            self._l2Cl = self.raw_table[self.mode].values

            if self.unit == "K^2.sr":
                self._l2Cl /= 1.0e12

            if self.unit in ["MJy^2/sr", "Jy^2/sr"]:
                self._l2Cl /= 1.0e12  # from uK^2.sr to K^2.sr
                self._l2Cl *= self.K2Jy[str(self.freq)] ** 2

            if self.unit == "MJy^2/sr":
                self._l2Cl /= 1.0e12

        return self._l2Cl

    @property
    def Cl(self):
        if self._Cl is None:
            self._Cl = self.l2Cl / self.l_scaling

        return self._Cl


class Planck18Data(CMBxCMB):
    """Planck PR3 CMB TT power spectrum, as measured on the data.
    """

    def __init__(self, freq=None, mode="TT", unit="uK^2.sr"):
        super().__init__(freq=freq, mode=mode, unit=unit)

    # Properties
    ############
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = pd.read_fwf(
                os.path.join(
                    P.PACKAGE_DIR,
                    "resources/cmbxcmb/COM_PowerSpect_CMB-TT-full_R3.01.txt",
                ),
                comment="#",
            )

        return self._raw_table

    @property
    def l(self):
        if self._l is None:
            self._l = self.raw_table["l"]
        return self._l

    @property
    def l_scaling(self):
        return self.l * (self.l + 1) / 2.0 / np.pi

    @property
    def l2Cl(self):
        """l(l+1)/2pi * Cl"""
        if self._l2Cl is None:
            # Native unit is uK^2/sr
            self._l2Cl = self.raw_table["Dl"]

            if self.unit == "K^2.sr":
                self._l2Cl /= 1.0e12

            if self.unit in ["MJy^2/sr", "Jy^2/sr"]:
                self._l2Cl /= 1.0e12  # from uK^2.sr to K^2.sr
                self._l2Cl *= self.K2Jy[str(self.freq)] ** 2

            if self.unit == "MJy^2/sr":
                self._l2Cl /= 1.0e12

        return self._l2Cl

    @property
    def dl2Cl(self):
        if self._dl2Cl is None:
            # Native unit is uK^2/sr
            self._dl2Cl = np.row_stack((self.raw_table["-dDl"], self.raw_table["+dDl"]))

            if self.unit == "K^2.sr":
                self._dl2Cl /= 1.0e12

            if self.unit in ["MJy^2/sr", "Jy^2/sr"]:
                self._dl2Cl /= 1.0e12  # from uK^2.sr to K^2.sr
                self._dl2Cl *= self.K2Jy[str(self.freq)] ** 2

            if self.unit == "MJy^2/sr":
                self._dl2Cl /= 1.0e12

        return self._dl2Cl

    @property
    def Cl(self):
        if self._Cl is None:
            self._Cl = self.l2Cl / self.l_scaling

        return self._Cl

    @property
    def dCl(self):
        if self._dCl is None:
            self._dCl = self.dl2Cl / self.l_scaling

        return self._dCl
