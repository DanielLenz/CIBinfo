import numpy as np

# Planck calibration correction to compare PR1 and PR2
# The convention is that PR2 = PR1 / C for the maps
# For powerspectra, the conversion factor needs to be applied
# once for each frequency
PLANCK_PR1_CALCORR = {
    "100": 0.994,
    "217": 0.993,
    "353": 0.977,
    "545": 1.018,
    "857": 1.033,
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
