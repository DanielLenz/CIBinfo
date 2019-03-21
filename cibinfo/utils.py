import numpy as np


def adjust_cl_length(cl, lmax=None, nside=None):
    """Many functions requires the Cl to have a length of exactly
    lmax or of 3 * nside. This function returns a Cl that is either cut
    or extended to fit this need.

    If the input Cl is too long, it is simply truncated.

    If the input Cl is too short, then the last element is
    simply repeated until the length is 3*nside or lmax.

    Parameters
    ----------
    cl: Input Cl, 1D np.ndarray
    nside: int, Valid HEALPix nside
    lmax: int, lmax
    """
    if lmax and nside:
        raise RuntimeError("Must provide nside OR lmax")

    if lmax is None:
        lmax = 3 * nside
    # > 0 if too short, < 0 if too long
    len_diff = lmax - cl.shape[0]

    # Just return for perfect match
    if len_diff == 0:
        return cl

    # Extend if too short
    if len_diff > 0:
        cl = np.concatenate((cl, np.repeat(cl[-1], repeats=len_diff)))

    # Cut if too long
    else:
        cl = cl[:lmax]

    return cl


# Planck calibration correction to compare PR1 and PR2
# The convention is that PR2 = PR1 / C for the maps
# For powerspectra, the conversion factor needs to be applied
# once for each frequency
# PLANCK_PR1PR2_CALCORR = {
#     "100": 0.994,
#     "217": 0.993,
#     "353": 0.977,
#     "545": 1.018,
#     "857": 1.033,
# }

# Planck calibration correction to compare PR1 and PR3
# The convention is that PR3 = PR1 / C for the maps
# For powerspectra, the conversion factor needs to be applied
# once for each frequency

PLANCK_PR1PR3_CALCORR = {
    "217": 1.00313,
    "353": 1.00789,
    "545": 1.00978,
    "857": 1.03044,
}

def phi2kappa(ells):
    """Converts the lensing potential to the lensing convergence. Needs to
    be applied in quadrature to power spectra (i.e. Cl^PP -> Cl^KK), and only
    once for a_LM."""

    return 0.5 * ells * (ells + 1.)


def kappa2phi(ells):
    """Converts the lensing convergence to the lensing potential. Needs to
    be applied in quadrature to power spectra (i.e. Cl^KK -> Cl^PP), and only
    once for a_LM."""

    return 1./phi2kappa(ells)
