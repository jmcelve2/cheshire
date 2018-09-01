
import numpy as np
from cheshire.Potential import *


class TestInstantiation(object):
    def test_potential(self):
        params = dict(potential=np.array([0, 0]))
        Potential(params=params)
        assert True

    def test_iw1d(self):
        InfiniteWell1D()
        assert True

    def test_iw2d(self):
        InfiniteWell2D()
        assert True

    def test_sho2d(self):
        SimpleHarmonicOscillator2D()
        assert True

    def test_dig2d(self):
        DoubleInvertedGaussian2D()
        assert True

    def test_rand2d(self):
        Random2D()
        assert True

    def test_coulomb(self):
        Coulomb2D()
        assert True
