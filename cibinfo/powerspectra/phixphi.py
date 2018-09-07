from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
import pandas as pd

from .. import this_project as P

__all__ = [
    'Planck15Kappa',
    'Planck15Phi',
    'Planck18Kappa',
    ]


class PhixPhi():

    _l = None  # multipole
    _Cl = None  # angular power
    _Nl = None  # noise power spectrum

    _df = None  # raw table, taken from publications, emails, etc

    def __init__(self):
        pass

    # Properties
    ############
    @property
    def l(self):
        if self._l is None:
            self._l = self.df['ell'].values.copy()
        return self._l

    @property
    def Cl(self):
        return None


class Planck15Kappa(PhixPhi):
    """Contains information on the CMB lensing power spectrum from the Planck
    (2015) PR2 release (https://arxiv.org/abs/1502.01591). Data are taken from
    https://wiki.cosmos.esa.int/planckpla2015/index.php/Specially_processed_maps#2015_Lensing_map
    and extrapolated for the multipoles 0-7.
    This class describes the lensing convergence $\kappa$, which is related
    to the lensing potential $\phi$ via
    $$ \kappa_{lm} = 0.5(\ell(\ell+1)) \phi_{lm} $$
    and
    $$ C_{\ell}^{\kappa\kappa} = [0.5(\ell(\ell+1))]^2 C_{\ell}^{\phi\phi} $$
    """

    def __init__(self):
        super(Planck15Kappa, self).__init__()

    # Properties
    ############
    @property
    def df(self):
        if self._df is None:
            self._df = pd.read_csv(os.path.join(
                P.PACKAGE_DIR,
                'resources/phixphi/Planck15_kappa.csv'),
                comment='#')
        return self._df

    @property
    def Cl(self):
        if self._Cl is None:
            self._Cl = self.df['Cl'].values.copy()
        return self._Cl

    @property
    def Nl(self):
        if self._Nl is None:
            self._Nl = self.df['Nl'].values.copy()
        return self._Nl


class Planck15Phi(PhixPhi):
    """Contains information on the CMB lensing power spectrum from the Planck
    (2015) PR2 release (https://arxiv.org/abs/1502.01591). Data are taken from
    https://wiki.cosmos.esa.int/planckpla2015/index.php/Specially_processed_maps#2015_Lensing_map
    and extrapolated for the multipoles 0-7.
    This class describes the lensing potential $\phi$, which is related
    to the lensing convergence $\kappa$ via
    $$ \kappa_{lm} = 0.5(\ell(\ell+1)) \phi_{lm} $$
    and
    $$ C_{\ell}^{\kappa\kappa} = [0.5(\ell(\ell+1))]^2 C_{\ell}^{\phi\phi} $$
    """

    _Cl = None
    _l4Cl = None

    def __init__(self):
        super(Planck15Phi, self).__init__()

    # Properties
    ############
    @property
    def df(self):
        if self._df is None:
            self._df = pd.read_csv(os.path.join(
                P.PACKAGE_DIR,
                'resources/phixphi/Planck15_Phi.csv'),
                comment='#')
        return self._df

    @property
    def Cl(self):
        if self._Cl is None:
            self._Cl = self.df['Cl'].values.copy()
        return self._Cl

    @property
    def l4Cl(self):
        """Returns [l(l+1)]^2/2pi Cl, which is a common format
        for the lensing potential."""
        if self._l4Cl is None:
            self._l4Cl = (self.l * (self.l + 1.))**2 / 2. / np.pi * self.Cl
        return self._l4Cl


class Planck18Kappa(PhixPhi):
    """Contains information on the CMB lensing power spectrum from the Planck
    (2018) PR3 release. Data are taken from the Planck Legacy Archive. We
    are using the minimum-variance (MV) data here.
    This class describes the lensing convergence $\kappa$, which is related
    to the lensing potential $\phi$ via
    $$ \kappa_{lm} = 0.5(\ell(\ell+1)) \phi_{lm} $$
    and
    $$ C_{\ell}^{\kappa\kappa} = [0.5(\ell(\ell+1))]^2 C_{\ell}^{\phi\phi} $$
    """

    def __init__(self):
        super().__init__()

    # Properties
    ############
    @property
    def df(self):
        if self._df is None:
            d = np.loadtxt(os.path.join(
                P.PACKAGE_DIR,
                'resources/phixphi/Planck18_kappa.txt'),)

            self._df = pd.DataFrame(d, columns=['ell', 'Nl', 'Cl+Nl'])

        return self._df

    @property
    def Cl(self):
        if self._Cl is None:
            self._Cl = (self.df['Cl+Nl'] - self.df['Nl']).values
        return self._Cl

    @property
    def Nl(self):
        if self._Nl is None:
            self._Nl = self.df['Nl'].values.copy()
        return self._Nl