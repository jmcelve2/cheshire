
import numpy as np
from cheshire.Kinetic import *


class TestInstantiation(object):
    def test_kinetic(self):
        params = dict(kinetic=np.array([0, 0]))
        Kinetic(params=params)
        assert True

    def test_kinetic1d(self):
        Kinetic1D()
        assert True

    def test_kinetic2d(self):
        Kinetic2D()
        assert True
