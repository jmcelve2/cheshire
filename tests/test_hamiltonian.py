
from cheshire.Hamiltonian import Hamiltonian


class TestEndToEnd(object):

    def test_well1d(self):
        grid_params = dict(n_x=32, n_e=2, x_min=-20, x_max=20)
        pot_params = dict(c_x=0, l_x=5)
        Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        assert True

    def test_well2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(c_x=0, c_y=0, l_x=5, l_y=5)
        Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        assert True

    def test_sho2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(c_x=0, c_y=0, k_x=1, k_y=1)
        Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        assert True

    def test_dig2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(a1=2, a2=2, c_x1=-2, c_x2=2, c_y1=-2, c_y2=2, k_x1=2, k_x2=2, k_y1=2, k_y2=2)
        Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        assert True

    def test_rand2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(k=2, r=20, sig1=3, sig2=5, p=2)
        Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        assert True

    def test_coul2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(z=1, c_x=0, c_y=0, alpha=10**-9)
        Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        assert True


class TestSolve(object):

    def test_well1d(self):
        grid_params = dict(n_x=32, n_e=2, x_min=-20, x_max=20)
        pot_params = dict(c_x=0, l_x=5)
        h = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        h.solve()
        assert True

    def test_well2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(c_x=0, c_y=0, l_x=5, l_y=5)
        h = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        h.solve()
        assert True

    def test_sho2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(c_x=0, c_y=0, k_x=1, k_y=1)
        h = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        h.solve()
        assert True

    def test_dig2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(a1=2, a2=2, c_x1=-2, c_x2=2, c_y1=-2, c_y2=2, k_x1=2, k_x2=2, k_y1=2, k_y2=2)
        h = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        h.solve()
        assert True

    def test_rand2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(k=2, r=20, sig1=3, sig2=5, p=2)
        h = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        h.solve()
        assert True

    def test_coul2d(self):
        grid_params = dict(n_x=32, n_y=32, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20)
        pot_params = dict(z=1, c_x=0, c_y=0, alpha=10**-9)
        h = Hamiltonian(grid_params=grid_params, pot_params=pot_params)
        h.solve()
        assert True
