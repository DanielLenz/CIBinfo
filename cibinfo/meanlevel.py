from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

# From Planck (2011) XVIII, Table 6, Units are MJy/sr
# Tuples are (value, 1-sigma uncertainty)
Fixsen1998 = {
    '217': (5.4e-2, 1.7e-2),
    '353': (1.6e-1, 0.5e-1),
    '545': (3.7e-1, 1.1e-1),
    '857': (6.5e-1, 2.0e-1),
}

Gispert2000 = {
    '217': (3.4e-2, 1.1e-2),
    '353': (1.3e-1, 0.4e-1),
    '545': (3.7e-1, 1.2e-1),
    '857': (7.1e-1, 2.3e-1),
}

# From Planck (2016) VIII, Table 6, Units are MJy/sr
# Monopole error is ~20%
Planck2016 = {
    '100': (0.003, 0.0006),
    '143': (0.0079, 0.00158),
    '217': (0.033, 0.0066),
    '353': (0.13, 0.026),
    '545': (0.35, 0.07),
    '857': (0.64, 0.128)
}
