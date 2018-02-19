
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.utils.extmath import cartesian
from cheshire.Kinetic import *
from cheshire.Potential import *

# Hartree atomic units are used, so all of these are unity.
HBAR = 1
CHARGE_E = 1
MASS_E = 1
MASS = 1


class Hamiltonian(object):
    """
    The Hamiltonian class converts parameters into a Hamiltonian and allows users to numerically solve the Schrodinger
    equation using finite element analysis. The particular solver function implemented depends on the parameters passed
    into the constructor.

    Attributes:
        **potential (Potential)**: An object of class Potential containing an array of potential energy values and
            distance information.
    """

    def __init__(self, grid_params, pot_params):
        """
        Numerically solve the Schrodinger equation.

        Args:
            **grid_params (dict)**: Parameters specifying the form of the grid and the number of electrons.
            **pot_params (dict)**: Parameters specifying the form of the potential.
        """
        assert isinstance(grid_params, dict)
        assert isinstance(pot_params, dict)
        if 'n_x' not in grid_params.keys():
            raise AssertionError("You must specify at least 'n_x' (in the 1D case). If the 2D case is desired, both "
                                 "'n_x' and 'n_y' must be specified.")
        if 'x_min' not in grid_params.keys() or 'x_max' not in grid_params.keys():
            raise AssertionError("You must specify at least 'x_min' and 'x_max', the parameters that define the "
                                 "physical size of the grid (in units of a.u.)")
        if 'n_y' in grid_params.keys() and ('y_min' not in grid_params.keys() or 'y_max' not in grid_params.keys()):
            raise AssertionError("You must specify 'y_min' and 'y_max'.")
        if 'n_z' in grid_params.keys():
            raise AssertionError("3D systems are not currently supported.")
        if 'n_e' not in grid_params.keys():
            raise AssertionError("You must specify the number of electrons parameter, 'n_e', in the parameter "
                                 "dictionary.")
        assert isinstance(grid_params['n_e'], int)
        assert grid_params['n_e'] in [1, 2]
        if grid_params['n_e'] == 2 and 'n_y' in grid_params.keys():
            raise AssertionError("2D potentials do not currently support multiple electrons.")

        self.params = dict(**grid_params, **pot_params)
        self.kinetic = kinetic_factory(grid_params=grid_params)
        self.potential = potential_factory(grid_params=grid_params, pot_params=pot_params)

        self.x = self.potential.x
        if 'n_y' in grid_params:
            self.y = self.potential.y

    def solve(self, k=1):
        """
        Solver method which conditionally implements Schrodinger solvers based on the dimensionality of the problem.
        """
        potential = np.diag(self.potential.potential.flatten())*CHARGE_E
        kinetic = self.kinetic.kinetic
        hamiltonian = kinetic + potential

        if self.params['n_e'] == 1:
            energy, psi = eigs(hamiltonian, k=k, which="SM")

            # Drop the imaginary component (which is always 0)
            energy = np.real(energy)

            # Reformat the eigenstates so that they are returned in the shape of the potential
            psi = np.array([np.transpose(psi[:, i]).flatten().reshape((self.params['n_x'], self.params['n_y']))
                            for i in range(psi.shape[1])])

        if self.params['n_e'] == 2:
            # Reconstruct the physical coordinates of the grid. This doesn't have to
            # agree with the linspace used in generating the potential since only
            # relative distances matter.
            x = self.x

            # Calculate the interaction term and set v_ee = np.max(v) for diffs == 0
            rdiff = abs(np.diff(cartesian((x, x))))[:, 0]
            rdiff[rdiff == 0] = 1 / np.max(self.potential.potential)
            v_ee = np.diag(1 / rdiff)

            # Calculate the full Hamiltonian: H = h x I + I x h + v_ee
            hamiltonian = np.kron(hamiltonian, np.eye(self.params['n_x'])) + \
                          np.kron(np.eye(self.params['n_x']), hamiltonian)
            hamiltonian = hamiltonian + v_ee
            energy, psi = eigs(hamiltonian, k=k, which="SM")

            # Drop the imaginary component (which is always 0)
            energy = np.real(energy)

            # Reformat the eigenstates so that they are reformatted as x1 vs x2
            psi = np.array([np.transpose(psi[:, i]).flatten().reshape((self.params['n_x'], self.params['n_x']))
                            for i in range(psi.shape[1])])

        return energy, psi
