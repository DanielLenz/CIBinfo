import unittest

import numpy as np

from cibinfo.powerspectra import cibxcib as TT
from cibinfo.powerspectra import cibxphi as TP
from cibinfo.powerspectra import phixphi as PP
from cibinfo.powerspectra import noise
from cibinfo import this_project as P


class TestCIBxCIB():
    freq = 545

    def test_Planck14_data(self):
        p14_data = TT.Planck14Data(self.freq, unit='Jy^2/sr')
        assert p14_data.l[0] == 50.056
        assert p14_data.Cl[0] == 107090

        np.testing.assert_almost_equal(
            p14_data.Cl[1],
            (
                p14_data.Jy2K[str(self.freq)] * p14_data.K2Jy[str(self.freq)] *
                p14_data.Cl[1])
        )

    def test_Planck14_model(self):
        p14_model = TT.Planck14Model(self.freq, unit='Jy^2/sr')
        assert p14_model.l[0] == 1
        assert p14_model.Cl[0] == 9204.2

        np.testing.assert_almost_equal(
            p14_model.Cl[1],
            (
                p14_model.Jy2K[str(self.freq)] *
                p14_model.K2Jy[str(self.freq)] * p14_model.Cl[1])
        )

    def test_Maniyar18_model(self):
        maniyar18_model = TT.Maniyar18Model(self.freq, unit='Jy^2/sr')
        assert maniyar18_model.l[0] == 2.
        np.testing.assert_almost_equal(
            maniyar18_model.Cl[0],
            2.762462665999e+04)

        np.testing.assert_almost_equal(
            maniyar18_model.Cl[1],
            (
                maniyar18_model.Jy2K[str(self.freq)] *
                maniyar18_model.K2Jy[str(self.freq)] * maniyar18_model.Cl[1])
        )


class TestCIBxPhi():
    freq = 353

    def test_Planck13_data(self):
        p13_data = TP.Planck13Data(self.freq, unit='uK.sr')
        np.testing.assert_almost_equal(
            p13_data.l[0],
            163.333328333)
        np.testing.assert_almost_equal(
            p13_data.l3Cl[0],
            0.029015585056)

        np.testing.assert_almost_equal(
            p13_data.Cl[1],
            (
                p13_data.Jy2K[str(self.freq)] * p13_data.K2Jy[str(self.freq)] *
                p13_data.Cl[1])
        )

        np.testing.assert_almost_equal(
            p13_data.l3Cl[2],
            p13_data.Cl[2] * p13_data.l[2]**3)

    def test_Planck13_model(self):
        p13_model = TP.Planck13Model(self.freq, unit='Jy')
        assert p13_model.l[0] == 1
        assert p13_model.Cl[0] == 0.0151478

        np.testing.assert_almost_equal(
            p13_model.Cl[1],
            (
                p13_model.Jy2K[str(self.freq)] *
                p13_model.K2Jy[str(self.freq)] * p13_model.Cl[1])
        )

    def test_Maniyar18_model(self):
        maniyar18_model = TP.Maniyar18Model(self.freq, unit='Jy')
        assert maniyar18_model.l[0] == 1.
        np.testing.assert_almost_equal(
            maniyar18_model.Cl[0],
            1.376221377266412249e-02)

        np.testing.assert_almost_equal(
            maniyar18_model.Cl[1],
            (
                maniyar18_model.Jy2K[str(self.freq)] *
                maniyar18_model.K2Jy[str(self.freq)] * maniyar18_model.Cl[1])
        )


class TestPhixPhi():
    def test_Planck15_kappa(self):
        p15_kappa = PP.Planck15Kappa()

        np.testing.assert_almost_equal(
            p15_kappa.l[3],
            3.)

        np.testing.assert_almost_equal(
            p15_kappa.Cl[0],
            1.029e-07)

    def test_Planck15_phi(self):
        p15_phi = PP.Planck15Phi()

        np.testing.assert_almost_equal(
            p15_phi.l[2],
            2.)

        np.testing.assert_almost_equal(
            p15_phi.Cl[2],
            1.2838e-08)

class TestNoise(unittest.TestCase):
    ...
