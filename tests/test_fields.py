import unittest

import numpy as np
from numpy import testing as npt

from cibinfo import fields

class TestSimCl(unittest.TestCase):
    def setUp(self):
        self.ref = fields.SimCl(freq='353', unit='MJy^2/sr')
    
    def test_units(self):
        with self.assertRaises(ValueError):
            fields.SimCl(freq='545', unit='Nonsense')

        units = ['MJy^2/sr', 'Jy^2/sr', 'K^2.sr', 'uK^2.sr']
        for unit in units:
            self.assertEqual(unit, fields.SimCl('545', unit).unit)
        
    def test_freqs(self):
        with self.assertRaises(ValueError):
            fields.SimCl(freq='999')
        with self.assertRaises(TypeError):
            fields.SimCl(freq=353)

        freqs = ['143', '217', '353', '545', '857']
        for freq in freqs:
            self.assertEqual(freq, fields.SimCl(freq).freq)

    def test_shapes(self):
        lengths = np.array([
            len(self.ref.tt),
            len(self.ref.cc),
            len(self.ref.kk),
            len(self.ref.tk),
            len(self.ref.noise_tt),
            len(self.ref.noise_kk),
        ])

        self.assertTrue((lengths == self.ref.lmax_sim).all())


class TestBaseField(unittest.TestCase):
    def setUp(self):
        self.field = fields.BaseField(nside=2048, lmax=2000)

    def test_lmax(self):
        # Base case
        lmax = 1024
        field = fields.BaseField(lmax=lmax)
        self.assertEqual(field.lmax, lmax)

        # Low nside
        nside = 256
        field = fields.BaseField(nside=nside)
        self.assertEqual(field.lmax, 3*nside)

        # High nside
        nside = 2048
        field = fields.BaseField(nside=nside)
        self.assertEqual(field.lmax, 2040)

    def test_nside(self):
        # Base case
        nside = 1024
        self.assertEqual(nside, fields.BaseField(nside).nside)

        # Invalid nside
        with self.assertRaises(ValueError):
            field = fields.BaseField(1000)

class TestField(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_cl(self):
        # Invalid type
        with self.assertRaises(TypeError):
            fields.Field(cl=1.)

        # Invalid shape
        with self.assertRaises(ValueError):
            fields.Field(cl=np.arange(10, dtype=float)[:, None])

        cl = np.random.random(size=100)
        npt.assert_array_almost_equal(cl, fields.Field(cl=cl).cl)

    def test_observe(self):
        cl = np.random.random(1000)
        f = fields.Field(cl, nside=256, lmax=512)
        f.observe(beam=None, pixwin=True)
        f.observe(beam=1., pixwin=False)
        f.observe(beam=np.linspace(1., 0, 1000), pixwin=True)