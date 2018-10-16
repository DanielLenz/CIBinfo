import numpy as np

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
