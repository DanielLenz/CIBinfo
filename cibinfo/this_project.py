from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Conversion factors from K_CMB to Jy/sr for the Planck bands
# Taken from Planck (2014 XXX), Table 2
K2Jy = {
    '100': 244.1e6,
    '143': 371.74e6,
    '217': 483.690e6,
    '353': 287.450e6,
    '545': 58.04e6,
    '857': 2.27e6, }

Jy2K = {k: 1. / v for k, v in iter(K2Jy.items())}
