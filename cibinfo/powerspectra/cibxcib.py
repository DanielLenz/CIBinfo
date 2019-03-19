from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Optional

import numpy as np
import os

from astropy.io import fits
from astropy.table import Table

from .. import this_project as P
from .. import utils as ut

__all__ = ["Planck14Data", "Planck14DataAlt", "Planck14Model", "Maniyar18Model"]


class CIBxCIB:

    _l = None  # multipole
    _Cl = None  # angular power
    _dCl = None  # uncertainty on the angular power
    _Dl = None  # l(l+1)/2pi * Cl
    _dDl = None  # uncertainty on the angular power
    _l3Cl = None  # l**3 * Cl

    _S = None  # shot noise level
    _dS = None  # uncertainty on the shot noise

    _raw_table = None  # raw table, taken from publications, emails, etc

    def __init__(self, freq1, freq2=None, unit="Jy^2/sr"):
        if unit not in ["Jy^2/sr", "MJy^2/sr", "K^2.sr", "uK^2.sr"]:
            raise ValueError(
                'Unit must be either "Jy^2/sr", ' '"MJy^2/sr", "uK^2.sr" or "K^2.sr"'
            )
        self.unit = unit

        self.freq1 = self.freq2int(freq1)
        if freq2 is None:
            self.freq2 = freq1
        else:
            self.freq2 = self.freq2int(freq2)

    # Methods
    #########
    def freq2int(self, freq):
        if isinstance(freq, str):
            return int(freq)
        else:
            if isinstance(freq, int):
                return freq
            else:
                raise TypeError("freq must be int or str")

    # Properties
    ############
    @property
    def freqstr(self):
        self._freqstr = "x".join(
            (
                str(max([int(self.freq1), int(self.freq2)])),
                str(min([int(self.freq1), int(self.freq2)])),
            )
        )
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
            self._Dl = self.l * (self.l + 1.0) / 2.0 / np.pi * self.Cl
        return self._Dl

    @property
    def l3Cl(self):
        if self._l3Cl is None:
            self._l3Cl = self.l ** 3 * self.Cl
        return self._l3Cl

    @property
    def Jy2K(self):
        self._Jy2K = P.Jy2K
        return self._Jy2K

    @property
    def K2Jy(self):
        self._K2Jy = P.K2Jy
        return self._K2Jy


class Planck14DataAlt(CIBxCIB):
    def __init__(self, freq1, freq2=None, unit="Jy^2/sr"):
        super(Planck14Data, self).__init__(freq1, freq2=freq2, unit=unit)

        self.Cl_contains_SN = True

    # Properties
    ############
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = np.loadtxt(
                os.path.join(P.PACKAGE_DIR, "resources/cibxcib/Planck14_data.txt")
            )
        return self._raw_table

    @property
    def Cl(self):
        if self._Cl is None:
            # Native unit is Jy^2/sr
            self._Cl = self.raw_table[:, self._freq2col(self.freqstr)].copy()

            # We add the shot noise, which is also in Jy^2/sr
            self._Cl += self.S

            if self.unit in ["K^2.sr", "uK^2.sr"]:
                self._Cl *= self.Jy2K[str(self.freq1)] * self.Jy2K[str(self.freq2)]

                if self.unit == "uK^2.sr":
                    self._Cl *= 1.0e12

            if self.unit == "MJy^2/sr":
                self._Cl /= 1.0e12

        return self._Cl

    @property
    def S(self):
        """
        Shot noise, units are Jy^2/sr. Taken from
        Planck (2014 XXX)
        """
        if self._S is None:
            self._S = {
                "857x857": 5364,
                "545x545": 1690,
                "353x353": 262,
                "217x217": 21,
                "3000x3000": 9585,
                "857x545": 2702,
                "857x353": 953,
                "857x217": 181,
                "545x353": 626,
                "545x217": 121,
                "353x217": 54,
                "3000x857": 4158,
                "3000x545": 1449,
                "3000x353": 411,
                "3000x217": 95,
            }

            if self.unit in ["K^2.sr", "uK^2.sr"]:
                self._S *= self.Jy2K[str(self.freq1)] * self.Jy2K[str(self.freq2)]

                if self.unit == "uK^2.sr":
                    self._S *= 1.0e12

            if self.unit == "MJy^2/sr":
                self._S /= 1.0e12

        return self._S[self.freqstr]

    @property
    def dS(self):
        """
        Shot noise
        """
        if self._dS is None:
            self._dS = {
                "857x857": 343,
                "545x545": 45,
                "353x353": 8,
                "217x217": 2,
                "3000x3000": 1090,
                "857x545": 124,
                "857x353": 54,
                "857x217": 6,
                "545x353": 19,
                "545x217": 6,
                "353x217": 3,
                "3000x857": 443,
                "3000x545": 176,
                "3000x353": 48,
                "3000x217": 11,
            }
            if self.unit in ["K^2.sr", "uK^2.sr"]:
                self._dS *= self.Jy2K[str(self.freq1)] * self.Jy2K[str(self.freq2)]
                if self.unit == "uK^2.sr":
                    self._dS *= 1.0e12

            if self.unit == "MJy^2/sr":
                self._dS /= 1.0e12

        return self._dS[self.freqstr]

    # Methods
    #########
    def _freq2col(self, freqstr):
        mapping = {
            "857x857": 1,
            "545x545": 2,
            "353x353": 3,
            "217x217": 4,
            "3000x3000": 5,
            "857x545": 6,
            "857x353": 7,
            "857x217": 8,
            "545x353": 9,
            "545x217": 10,
            "353x217": 11,
            "3000x857": 12,
            "3000x545": 13,
            "3000x353": 14,
            "3000x217": 15,
            "100x100": 16,
            "857x100": 17,
            "545x100": 18,
            "353x100": 19,
            "217x100": 20,
            "143x143": 21,
            "857x143": 22,
            "545x143": 23,
            "353x143": 24,
            "217x143": 25,
            "143x100": 26,
        }

        return mapping[self.freqstr]


class Planck14Model(CIBxCIB):
    def __init__(self, freq1, freq2=None, unit="Jy^2/sr"):
        super(Planck14Model, self).__init__(freq1, freq2=freq2, unit=unit)

        self.Cl_contains_SN = True

    # Properties
    ############
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = np.loadtxt(
                os.path.join(P.PACKAGE_DIR, "resources/cibxcib/Planck14_model.txt")
            )
        return self._raw_table

    @property
    def Cl(self):
        if self._Cl is None:
            # native unit is be Jy^2/sr
            self._Cl = self.raw_table[:, self._freq2col(self.freqstr)].copy()

            # Possibly convert the units
            if self.unit in ["K^2.sr", "uK^2.sr"]:
                self._Cl *= self.Jy2K[str(self.freq1)] * self.Jy2K[str(self.freq2)]
                if self.unit == ["uK^2.sr"]:
                    self._Cl *= 1.0e12

            if self.unit == "MJy^2/sr":
                self._Cl /= 1.0e12

            # Apply the correction factor to the PR1 calibration
            self._Cl /= (
                ut.PLANCK_PR1_CALCORR[str(self.freq1)]
                * ut.PLANCK_PR1_CALCORR[str(self.freq2)]
            )
        return self._Cl

    # Methods
    #########
    def _freq2col(self, freqstr):
        # l, 353x353, 353x545, 353x857, 545x545, 545x857, 857x857
        mapping = {
            "353x353": 1,
            "545x353": 2,
            "857x353": 3,
            "545x545": 4,
            "857x545": 5,
            "857x857": 6,
        }

        return mapping[self.freqstr]


class Maniyar18Model(CIBxCIB):
    def __init__(self, freq1, freq2=None, unit="Jy^2/sr"):
        super(Maniyar18Model, self).__init__(freq1, freq2=freq2, unit=unit)

        self.Cl_contains_SN = True
        self.Cl_contains_1halo = True

    # Properties
    ############
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = fits.open(
                os.path.join(
                    P.PACKAGE_DIR, "resources/cibxcib/Maniyar18_model_crosspowers.dat"
                )
            )
        return self._raw_table

    @property
    def l(self):
        if self._l is None:
            self._l = self.raw_table[1].data
        return self._l

    @property
    def Cl(self):
        if self._Cl is None:
            # native unit is Jy^2/sr
            self._Cl = self.raw_table[0].data[
                self._freq2col(str(self.freq1)), self._freq2col(str(self.freq2))
            ]

            # Possibly convert the units
            if self.unit in ["K^2.sr", "uK^2.sr"]:
                self._Cl *= self.Jy2K[str(self.freq1)] * self.Jy2K[str(self.freq2)]
                if self.unit == ["uK^2.sr"]:
                    self._Cl *= 1.0e12

            if self.unit == "MJy^2/sr":
                self._Cl /= 1.0e12

        return self._Cl

    # Methods
    #########
    def _freq2col(self, freq):
        # 217x217, 353x353, 545x545, 857x857, 3000x3000
        mapping = {
            "100": 0,
            "143": 1,
            "217": 2,
            "353": 3,
            "545": 4,
            "857": 5,
            "3000": 6,
        }

        return mapping[freq]


class Planck14Data(CIBxCIB):
    def __init__(self, freq1, freq2=None, unit="Jy^2/sr"):
        super(Planck14Data, self).__init__(freq1, freq2=freq2, unit=unit)
        self.Cl_contains_SN = True

    # Properties
    ############
    @property
    def raw_table(self):
        if self._raw_table is None:
            self._raw_table = Table.read(
                os.path.join(
                    P.PACKAGE_DIR, "resources/cibxcib/Planck14_data_frompaper.txt"
                ),
                format="csv",
                delimiter=";",
            )
        return self._raw_table

    @property
    def l(self):
        if self._l is None:
            self._l = self.raw_table["ell"].data.data

        return self._l

    @property
    def Cl(self):
        if self._Cl is None:
            # Native unit is Jy^2/sr
            self._Cl = self.raw_table[f"{self.freq1}x{self.freq2}"].data.data

            # We add the shot noise, which is also in Jy^2/sr
            # self._Cl += self.S

            if self.unit in ["K^2.sr", "uK^2.sr"]:
                self._Cl *= self.Jy2K[str(self.freq1)] * self.Jy2K[str(self.freq2)]

                if self.unit == "uK^2.sr":
                    self._Cl *= 1.0e12

            if self.unit == "MJy^2/sr":
                self._Cl /= 1.0e12

            # Apply the correction factor to the PR1 calibration
            self._Cl /= (
                ut.PLANCK_PR1_CALCORR[str(self.freq1)]
                * ut.PLANCK_PR1_CALCORR[str(self.freq2)]
            )

        return self._Cl

    @property
    def dCl(self):
        if self._dCl is None:
            # Native unit is Jy^2/sr
            self._dCl = self.raw_table[f"d{self.freq1}x{self.freq2}"].data.data

            if self.unit in ["K^2.sr", "uK^2.sr"]:
                self._dCl *= self.Jy2K[str(self.freq1)] * self.Jy2K[str(self.freq2)]

                if self.unit == "uK^2.sr":
                    self._dCl *= 1.0e12

            if self.unit == "MJy^2/sr":
                self._dCl /= 1.0e12

            # Apply the correction factor to the PR1 calibration
            self._dCl /= (
                ut.PLANCK_PR1_CALCORR[str(self.freq1)]
                * ut.PLANCK_PR1_CALCORR[str(self.freq2)]
            )

        return self._dCl

    @property
    def dDl(self):
        if self._dDl is None:
            self._dDl = self.l * (self.l + 1.0) / 2.0 / np.pi * self.dCl
        return self._dDl


class Mak18(CIBxCIB):
    def __init__(
        self,
        freq1: str,
        freq2: Optional[str] = None,
        lmax: Optional[int] = None,
        unit: str = "uK2.sr",
        mask="mask30",
    ):
        self.unit = unit
        self._Cl = None
        self._Dl = None

        if lmax is None:
            self.lmax = 3000
        else:
            self.lmax = lmax

        self.freq1 = freq1
        if freq2 is None:
            self.freq2 = freq1
        else:
            self.freq2 = freq2

        self.mask = mask

    @property
    def l(self):
        if self._l is None:
            self._l = np.arange(self.lmax)
        return self._l

    @staticmethod
    def model(ells, A_cib, A_ps, gamma):
        """
        Returns Dl^total = Dl^CIB + Dl^PS
        The Dl^PS are given as value at ell=2000, hence we need to convert this back to Cls
        and then apply the appropriate scaling.
        """
        return (
            A_cib * np.power(ells / 2000.0, gamma)
            + A_ps / 2000 / 2001 * ells * ells
            + 1.0
        )

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, val):
        if not hasattr(self, "_mask"):
            allowed = ["mask30", "mask40", "mask50"]
            if val not in allowed:
                raise ValueError(f"mask must be in {allowed}")
            self._mask = val
        else:
            raise RuntimeError("mask can only be set once during initialization.")

    @staticmethod
    def get_rho_cib(freq1, freq2):
        d = {
            "353x353": 1.0,
            "545x545": 1.0,
            "857x857": 1.0,
            "353x545": 0.975,
            "353x857": 0.892,
            "545x857": 0.949,
        }

        return d[f"{freq1}x{freq2}"]

    @staticmethod
    def get_rho_ps(freq1, freq2):
        d = {
            "353x353": 1.0,
            "545x545": 1.0,
            "857x857": 1.0,
            "353x545": 0.98,
            "353x857": 0.86,
            "545x857": 0.97,
        }

        return d[f"{freq1}x{freq2}"]

    @property
    def _raw_data(self):
        # Define mask30
        mask30 = {
            "gamma": 0.51,
            "cib": {"353": 2.5e3, "545": 4.5e5, "857": 1.09e9},
            "ps": {"353": 2.15e3, "545": 3.54e5, "857": 7.2e8},
            "cal": {"353": 1.0, "545": 1.05, "857": 1.01},
        }

        # Define mask40
        mask40 = {
            "gamma": 0.53,
            "cib": {"353": 2.56e3, "545": 4.47e5, "857": 1.09e9},
            "ps": {"353": 2.1e3, "545": 3.42e5, "857": 7.34e8},
            "cal": {"353": 1.0, "545": 1.03, "857": 1.01},
        }

        d = dict(
            mask30=mask30,
            mask40=mask40,
            #                mask50=mask50,
        )

        return d

    def calibration(self, freq1: str, freq2: str):
        raw_data = self._raw_data[self.mask]
        cal1, cal2 = raw_data["cal"][freq1], raw_data["cal"][freq2]

        return cal1 * cal2

    def get_model_parameters(self):
        raw_data = self._raw_data[self.mask]

        # Correlation coefficients
        rho_cib = self.get_rho_cib(self.freq1, self.freq2)
        rho_ps = self.get_rho_ps(self.freq1, self.freq2)

        # Amplitudes
        gamma = raw_data["gamma"]

        A_cib = rho_cib * np.sqrt(
            raw_data["cib"][self.freq1] * raw_data["cib"][self.freq2]
        )

        A_ps = rho_ps * np.sqrt(raw_data["ps"][self.freq1] * raw_data["ps"][self.freq2])

        return gamma, A_cib, A_ps

    @property
    def Cl(self):
        return self.Dl / (self.l * (self.l + 1)) * 2.0 * np.pi

    @property
    def Dl(self):
        if self._Dl is None:
            gamma, A_cib, A_ps = self.get_model_parameters()
            self._Dl = self.model(self.l, A_cib, A_ps, gamma)
            self._Dl /= self.calibration(self.freq1, self.freq2)

            if self.unit == "K^2.sr":
                self._Dl /= 1.0e12
            if self.unit == "MJy^2/sr":
                self._Dl *= 1.0e-12 * (self.K2MJy[self.freq1] * self.K2MJy[self.freq2])
            if self.unit == "Jy^2/sr":
                self._Dl *= self.K2MJy[self.freq1] * self.K2MJy[self.freq2]
        return self._Dl
