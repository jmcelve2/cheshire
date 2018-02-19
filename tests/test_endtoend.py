
from cheshire.ParamSampler import *
from cheshire.Hamiltonian import Hamiltonian


class TestHamiltonian(object):

    def test_hamiltonian_iw1d(self):
        grid_params = dict(n_x=32, n_e=2, x_min=-20, x_max=20)
        pot_params = dict(c_x=0, l_x=5)
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True

    def test_hamiltonian_iw2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(c_x=0, c_y=0, l_x=5, l_y=5)
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True

    def test_hamiltonian_sho2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(c_x=0, c_y=0, k_x=2, k_y=2)
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True

    def test_hamiltonian_dig2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(a1=2, a2=2, c_x1=-2, c_x2=2, c_y1=-2, c_y2=2, k_x1=2, k_x2=2, k_y1=2, k_y2=2)
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True

    def test_hamiltonian_rand2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(k=5, r=40, sig1=8, sig2=13, p=2)
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True

    def test_hamiltonian_coul2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(z=1, c_x=0, c_y=0, alpha=10**-9)
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True


class TestHamiltonianSampler(object):

    def test_hamiltonian_iw1d(self):
        grid_params = dict(n_x=32, n_e=2, x_min=-20, x_max=20)
        sampler = InfiniteWell1DSampler(**grid_params)
        pot_params = sampler.sample_params()
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True

    def test_hamiltonian_iw2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        sampler = InfiniteWell2DSampler(**grid_params)
        pot_params = sampler.sample_params()
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True

    def test_hamiltonian_sho2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        sampler = SimpleHarmonicOscillator2DSampler(**grid_params)
        pot_params = sampler.sample_params()
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True

    def test_hamiltonian_dig2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        sampler = DoubleInvertedGaussian2DSampler(**grid_params)
        pot_params = sampler.sample_params()
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True

    def test_hamiltonian_rand2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        sampler = Random2DSampler(**grid_params)
        pot_params = sampler.sample_params()
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True

    def test_hamiltonian_coul2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        sampler = Coulomb2DSampler(**grid_params)
        pot_params = sampler.sample_params()
        hamiltonian = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        hamiltonian.solve()
        assert True
