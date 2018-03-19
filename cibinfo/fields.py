import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from .powerspectra import autopower as TT
from .powerspectra import crosslensing as TP
from .powerspectra import phiphi as PP
from . import meanlevel


def make_fields(frequency, nside, resolution):
    """
    Creates correlated large-scale structure fields such as the CIB and the CMB
    lensing, using their auto- and cross power spectra.

    Parameters
    ----------
    frequency : array-like
        Frequency for the CIB map, must be one of the Planck bands from 353
        to 857 GHz
    nside : int
        HEALPix nside
    resolution : float
        FWHM resolution of the maps in degree

    Returns
    -------
    fields, Cls : dict-like-like, dict-like
        Dictionary of the HEALPix maps (in MJy/sr for the CIB component),
        DataFrame of the auto and cross power spectrum
    """
    print('Generating LSS images...')

    fields = {}

    # CIBxPhi power spectra
    Cl_TP = TP.Model(frequency, unit='MJy').Cl

    # Lensing auto power spectrum
    Cl_PP = PP.P15Phi().Cl

    # CIB auto power
    # Auto
    Cl_TT = TT.PaoloModel(frequency, unit='MJy^2/sr').Cl

    # Determine maximum multipole for which all power spectra have information
    lmax_synalm = np.amin([
        Cl_TP.shape[0],
        Cl_TT.shape[0],
        Cl_PP.shape[0]])

    # Create a DataFrame to save all the power spectra
    d = {
        'ell': np.arange(lmax_synalm),
        'CIBxCIB': Cl_TT[:lmax_synalm],
        'PhixPhi': Cl_PP[:lmax_synalm],
        'CIBxPhi': Cl_TP[:lmax_synalm]}

    Cl_df = pd.DataFrame(d, index=None)

    # Generate the alm
    almTT, almPP = hp.synalm([
        Cl_TT[:lmax_synalm],
        Cl_TP[:lmax_synalm],
        Cl_PP[:lmax_synalm]])

    # Make the maps
    almkw = dict(pixwin=True, fwhm=np.radians(resolution), nside=nside)
    mapTT = hp.alm2map(almTT, **almkw)
    mapPP = hp.alm2map(almPP, nside=nside)

    # Set the ells
    ell = np.arange(lmax_synalm)

    fields['CIB'] = mapTT
    fields['Phi'] = mapPP

    return fields, Cl_df


def test():
    fields, Cl_df = make_fields('545', 256, 1.5)

    # mollview
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        figsize=(14, 8), nrows=2, ncols=2)

    plt.axes(ax1)
    hp.mollview(fields['CIB'], hold=True)

    plt.axes(ax2)
    hp.mollview(fields['Phi'], hold=True)

    # CIB x CIB
    ax3.loglog(Cl_df['ell'], Cl_df['CIBxCIB'])
    ax3.set_xlabel(r'$\ell$')
    ax3.set_ylabel(r'$C_{\ell}^{545}$')

    # CIB x Phi
    ax4.plot(Cl_df['ell'], Cl_df['ell']**3 * Cl_df['CIBxPhi'])
    ax4.set_xlabel(r'$\ell$')
    ax4.set_ylabel(r'$C_{\ell}^{\phi\times 545}$')

    plt.show()

    return fields, Cl_df


if __name__ == '__main__':
    test()
