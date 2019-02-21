"""Generates correlated cosmic fields in HEALPix, based on the auto and cross
angular power spectra.

"""

import os

import numpy as np
import healpy as hp
import pandas as pd

from . import utils as ut
from .powerspectra import (
    cibxcib as TT,
    cibxphi as TK,
    phixphi as KK,
    cmbxcmb as CC,
    noise,
)


class SimCl:
    """Handles power spectra for the CIB, CMB, and the lensing convergence kappa
    in a consistent manner. 
    
    Examples
    --------
    cls = SimCl(freq=545)
    >> cl.tt
    >> cl.ell
    >> cl.noise_tt
    >> cl.cc
    """

    _ell = None
    _ell_data = None

    _tt = None          # CIB Cl
    _tt_data = None     # CIB P14 Cl data
    _dtt_data = None    # CIB P14 Cl data
    _cc = None          # CMB Cl
    _kk = None          # Lensing convergence Cl
    _tk = None          # Cl^CIBxKappa

    _noise_tt = None  # Planck noise
    _noise_kk = None  # Lensing convergence noise

    def __init__(self, freq, unit="MJy^2/sr"):
        self.freq = freq
        self.unit = unit

    @staticmethod
    def autopower2crosspower_units(autopower_unit: str) -> str:
        """Mapping of units of the CIB auto power spectrum (e.g. MJy^2/sr) to
        units of the cross power spectrum with the lensing (e.g. MJy).
        """
        conv = {"MJy^2/sr": "MJy", "Jy^2/sr": "Jy", "uK^2.sr": "uK", "K^2.sr": "K"}

        return conv[autopower_unit]

    @property
    def lmax_sim(self):
        """The maximum ell of all the different power spectrum models used
        here."""
        return 2500

    @property
    def ell(self):
        if self._ell is None:
            self._ell = np.arange(self.lmax_sim)
        return self._ell

    @property
    def ell_data(self):
        if self._ell_data is None:
            self._ell_data = TT.Planck14Data(self.freq, unit=self.unit).l
        return self._ell_data

    @property
    def tt_data(self):
        if self._tt_data is None:
            self._tt_data = TT.Planck14Data(self.freq, unit=self.unit).Cl
        return self._tt_data

    @property
    def dtt_data(self):
        if self._dtt_data is None:
            self._dtt_data = TT.Planck14Data(self.freq, unit=self.unit).dCl
        return self._dtt_data

    @property
    def tt(self):
        if self._tt is None:
            self._tt = TT.Maniyar18Model(self.freq, unit=self.unit).Cl
            self._tt = np.concatenate(([self._tt[0], self._tt[0]], self._tt))
        return self._tt[: self.lmax_sim]

    @property
    def cc(self):
        if self._cc is None:
            self._cc = CC.Planck15Model(self.freq, unit=self.unit).Cl
            self._cc = np.concatenate(([self._cc[0], self._cc[0]], self._cc))
        return self._cc[: self.lmax_sim]

    @property
    def kk(self):
        if self._kk is None:
            self._kk = KK.Planck18Kappa().Cl[: self.lmax_sim]
        return self._kk

    @property
    def tk(self):
        if self._tk is None:
            self._tk = TK.Maniyar18Model(
                self.freq,
                unit=self.autopower2crosspower_units[self.unit]).Cl

            # The monopole is missing, so we simply add one by hand
            self._tk = np.concatenate(([self._tk[0]], self._tk))[: self.lmax_sim]

            # TK.Planck13Model() gives us the lensing potential Phi. We need
            # to convert this to the lensing convergence kappa.
            self._tk *= ut.phi2kappa(np.arange(self.lmax_sim))

        return self._tk

    @property
    def noise_tt(self):
        if self._noise_tt is None:
            self._noise_tt = noise.PlanckPR3(self.freq, unit=self.unit).Nl[
                : self.lmax_sim
            ]
        return self._noise_tt

    @property
    def noise_kk(self):
        if self._noise_kk is None:
            self._noise_kk = KK.Planck18Kappa().Nl[: self.lmax_sim]
        return self._noise_kk


class Field:
    """Base class for Gaussian random fields"""


    def __init__(self, nside=1024, lmax=None):
        _lmax = None
        self.nside = nside
        self.lmax = lmax

    @property
    def npix(self):
        return hp.nside2npix(self.nside)

    @property
    def lmax(self):
        return self._lmax

    @lmax.setter
    def lmax(self, value):
        if value is None:
            # ell=1998 is the upper limit for the simulated Cls
            # We use either that, or the typical 3*nside to simulate the
            # maps
            self._lmax = min(1998, 3 * self.nside)
        elif not isinstance(value, int):
            raise TypeError("lmax must be int or None.")
        else:
            self._lmax = value



# class CIB(Field):
#     """Generates random realizations of CIB fields, based on the
#     Maniyar+ (2018) model."""


#     def __init__(
#             self, freq, unit='MJy^2/sr', nside=1024, pixwin=False, fwhm=0.0,
#             add_noise=False, lmax=None):

#         super().__init__(nside=nside, lmax=lmax)

#         self.freq = freq
#         self.unit = unit
#         self.add_noise = add_noise

#         # Create Cls
#         self.sim_cl = SimCl(freq=freq, unit=unit)

#     def generate(self):
#         self.generate_signal()
#         self.generate_noise()

#         return self.hpxmap_X, self.hpxmap_Y

#     def generate_signal(self):
#         self.alm_X = hp.synalm(self.Cl_XX, new=True)

#     def generate_noise(self):
#         if self.add_noise:
#             self.noisemap_X = hp.synfast(
#                 self.Nl, nside=self.nside,
#                 pol=False)
#         else:
#             self.noisemap_X = np.zeros(self.npix)

#         if self.add_noise2:
#             self.noisemap_Y = hp.synfast(
#                 self.Nl_YY, nside=self.nside,
#                 pol=False)
#         else:
#             self.noisemap_Y = np.zeros(self.npix)

#     @property
#     def hpxmap_X(self):
#         self._hpxmap_X = self.noisemap_X + self.signalmap_X

#         return self._hpxmap_X

#     @property
#     def Cl_tot(self):
#         if self.add_noise:
#             self._Cl_tot = self.Cl + self.Nl
#         else:
#             self._Cl_tot = self.Cl.copy()

#         return self._Cl_tot[:self.lmax]

#     def generate(self):
#         """Generates a random realization of the CIB by updating the alm
#         and hpx_map properties."""

#         self._cib_alm = hp.synalm(self.Cl_tot, verbose=False, lmax=self.lmax)
#         self._cib_map = hp.alm2map(
#             self.cib_alm,
#             nside=self.nside,
#             pixwin=self.pixwin,
#             fwhm=np.radians(self.fwhm),
#             pol=False)

#         return self.cib_map

#     @property
#     def cib_alm(self):
#         return self._cib_alm

#     @property
#     def cib_map(self):
#         return self._cib_map


class CorrField(Field):


    def __init__(
        self,
        freq,
        unit="MJy^2/sr",
        nside=1024,
        pixwin1=False,
        fwhm1=0.0,
        add_noise1=False,
        pixwin2=False,
        fwhm2=0.0,
        add_noise2=False,
        lmax=None,
    ):

        _lmax = None
        _Cl_XX = None
        _Cl_YY = None
        _Nl_XX = None
        _Nl_YY = None
        _Cl_XY = None

        _hpxmap_X = None
        _hpxmap_Y = None
        _noisemap_X = None
        _noisemap_Y = None

        super().__init__(nside=nside, lmax=lmax)

        self.unit = unit
        self.pixwin1 = pixwin1
        self.pixwin2 = pixwin2
        self.fwhm1 = fwhm1
        self.fwhm2 = fwhm2
        self.add_noise1 = add_noise1
        self.add_noise2 = add_noise2

        # Initialize simulation with first realization
        self.generate()

    @property
    def Cl_XX(self):
        raise NotImplementedError()

    @property
    def Cl_YY(self):
        raise NotImplementedError()

    @property
    def Nl_XX(self):
        raise NotImplementedError()

    @property
    def Nl_YY(self):
        raise NotImplementedError()

    @property
    def Cl_XY(self):
        raise NotImplementedError()

    def generate(self):
        self.generate_signal()
        self.generate_noise()

        return self.hpxmap_X, self.hpxmap_Y

    def generate_signal(self):
        self.alm_X, self.alm_Y = hp.synalm(
            [self.Cl_XX, self.Cl_YY, self.Cl_XY], new=True
        )

    def generate_noise(self):
        if self.add_noise1:
            self.noisemap_X = hp.synfast(
                self.Nl_XX, nside=self.nside, pol=False, verbose=False
            )
        else:
            self.noisemap_X = np.zeros(self.npix)

        if self.add_noise2:
            self.noisemap_Y = hp.synfast(
                self.Nl_YY, nside=self.nside, pol=False, verbose=False
            )
        else:
            self.noisemap_Y = np.zeros(self.npix)

    @property
    def hpxmap_X(self):
        self._hpxmap_X = self.noisemap_X + self.signalmap_X

        return self._hpxmap_X

    @property
    def hpxmap_Y(self):
        self._hpxmap_Y = self.noisemap_Y + self.signalmap_Y

        return self._hpxmap_Y

    @property
    def signalmap_X(self):
        self._signalmap_X = hp.alm2map(
            self.alm_X,
            pixwin=self.pixwin1,
            fwhm=np.radians(self.fwhm1),
            nside=self.nside,
            pol=False,
            verbose=False,
        )

        return self._signalmap_X

    @property
    def signalmap_Y(self):
        self._signalmap_Y = hp.alm2map(
            self.alm_Y,
            pixwin=self.pixwin2,
            fwhm=np.radians(self.fwhm2),
            nside=self.nside,
            pol=False,
            verbose=False,
        )

        return self._signalmap_Y


class CIBxCIB(CorrField):
    _alm = None
    _signalmap = None

    """Generates two CIB fields with identical signal, but different noise
    properties."""

    def __init__(
        self,
        freq,
        unit="MJy^2/sr",
        nside=1024,
        pixwin=True,
        fwhm=0.0,
        add_noise=True,
        lmax=None,
    ):


        self.sim_cl = SimCl(freq=freq, unit=unit)

        super().__init__(
            freq=freq,
            unit=unit,
            nside=nside,
            pixwin1=pixwin,
            fwhm1=fwhm,
            pixwin2=pixwin,
            fwhm2=fwhm,
            add_noise1=add_noise,
            add_noise2=add_noise,
            lmax=lmax,
        )

        

    @property
    def Cl_XX(self):
        if self._Cl_XX is None:
            self._Cl_XX = self.sim_cl.tt.copy()

        return self._Cl_XX[: self.lmax]

    @property
    def Nl_XX(self):
        if self._Nl_XX is None:
            if self.add_noise1:
                self._Nl_XX = self.sim_cl.noise_tt[: self.lmax]
            else:
                self._Nl_XX = np.zeros(self.lmax)

        return self._Nl_XX

    @property
    def alm(self):
        return self._alm

    @alm.setter
    def alm(self, val):
        self._alm = val

    def generate_signal(self):
        self.alm = hp.synalm(self.Cl_XX)

    # def generate_noise(self):
    #     if self.add_noise1:
    #         self.noisemap_X = hp.synfast(self.Nl_XX, nside=self.nside, pol=False)
    #     else:
    #         self.noisemap_X = np.zeros(self.npix)

    #     if self.add_noise2:
    #         self.noisemap_Y = hp.synfast(self.Nl_XX, nside=self.nside, pol=False)
    #     else:
    #         self.noisemap_Y = np.zeros(self.npix)

    @property
    def hpxmap_X(self):
        self._hpxmap_X = self.noisemap_X + self.signalmap

        return self._hpxmap_X

    @property
    def hpxmap_Y(self):
        self._hpxmap_Y = self.noisemap_Y + self.signalmap

        return self._hpxmap_Y

    @property
    def signalmap(self):
        self._signalmap = hp.alm2map(
            self.alm,
            pixwin=self.pixwin1,
            fwhm=np.radians(self.fwhm1),
            nside=self.nside,
            pol=False,
        )

        return self._signalmap


class CIBxKappa(CorrField):
    """Generates correlated fields of CIB and lensing convergence kappa."""

    def __init__(
        self,
        freq,
        unit="MJy^2/sr",
        nside=1024,
        pixwin_cib=False,
        fwhm_cib=0.0,
        pixwin_lensing=False,
        fwhm_lensing=0.0,
        add_cib_noise=False,
        add_lensing_noise=False,
        lmax=None,
    ):

        super().__init__(
            freq=freq,
            unit=unit,
            nside=nside,
            pixwin1=pixwin_cib,
            fwhm1=fwhm_cib,
            pixwin2=pixwin_lensing,
            fwhm2=fwhm_lensing,
            add_noise1=add_cib_noise,
            add_noise2=add_lensing_noise,
            lmax=lmax,
        )

        self.sim_cl = SimCl(freq=freq, unit=unit)

    @property
    def Cl_XX(self):
        if self._Cl_XX is None:
            self._Cl_XX = self.sim_cl.tt.copy()

        return self._Cl_XX[: self.lmax]

    @property
    def Cl_YY(self):
        if self._Cl_YY is None:
            self._Cl_YY = self.sim_cl.kk.copy()

        return self._Cl_YY[: self.lmax]

    @property
    def Cl_XY(self):
        if self._Cl_XY is None:
            self._Cl_XY = self.sim_cl.tk.copy()

        return self._Cl_XY[: self.lmax]

    @property
    def Nl_XX(self):
        if self._Nl_XX is None:
            if self.add_noise1:
                self._Nl_XX = self.sim_cl.noise_tt[: self.lmax]
            else:
                self._Nl_XX = np.zeros(self.lmax)

        return self._Nl_XX

    @property
    def Nl_YY(self):
        if self._Nl_YY is None:
            if self.add_noise2:
                self._Nl_YY = self.sim_cl.noise_kk[: self.lmax]
            else:
                self._Nl_YY = np.zeros(self.lmax)

        return self._Nl_YY


def main():
    pass


if __name__ == "__main__":
    main()
