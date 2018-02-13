
import numpy as np
import numpy.matlib as np_matlib
from scipy.sparse.linalg import eigs
from sklearn.utils.extmath import cartesian

# Hartree atomic units are used, so all of these are unity.
hbar = 1
e = 1
me = 1
Mass = 1


class Schrodinger:
    """
    The Schrodinger class numerically solves the Schrodinger equation using
    finite element analysis. The particular solver function implemented
    depends on the dimensionality of the potential passed to the constructor.

    Note:
        Using a class structure ensures that the interface for each
        solver method doesn't change with the dimensionality of the
        potential. This was done for consistency-of-use between
        solver types.

    Attributes:
        **potential (Potential)**: An object of class Potential containing
            an array of potential energy values and distance information.
        **k (int)**: The number of eigenvalues and eigenvectors to return
            from the diagonalization process. Default is 1.
    """

    def __init__(self, potential, k=1):
        """
        Numerically solve the Schrodinger equation.

        Args:
            **potential (Potential)**: An object of class Potential containing
                an array of potential energy values and distance information.
            **k (int)**: The number of eigenvalues and eigenvectors to return
                from the diagonalization process. Default is 1.
        """
        assert isinstance(potential.potential, np.ndarray)
        if not hasattr(potential, 'dist'):
            raise AssertionError('The potential must possess a field "d" ' +
                                 'corresponding to the physical distance between ' +
                                 'points on the potential grid.')
        assert potential.dist > 0
        assert isinstance(k, int)
        assert k >= 1

        self.potential = potential
        self.k = k

    def solve(self):
        """
        Solver method which conditionally implements Schrodinger solvers
        based on the dimensionality of the problem.
        """
        if len(self.potential.potential.shape) == 1:
            energy, psi = __schrodinger1d__(potential=self.potential, k=self.k)
            return energy, psi
        if len(self.potential.potential.shape) == 2:
            energy, psi = __schrodinger2d__(potential=self.potential, k=self.k)
            return energy, psi


def __schrodinger1d__(potential, k=1):
    """
    Numerically solve the one-dimensional Schrodinger equation with two
    electrons.

    Args:
        **potential (np.ndarray)**: A grid of potential values.
        **k (int)**: The number of eigenvectors and eigenvalues to return.

    Returns:
        **E (numpy.array)**: An array of energy eigenvalues.
        **psi (numpy.array)**: An array of size (k, n_x*n_y) containing the
            amplitude values of the solution to the Schrodinger equation.
    """
    assert isinstance(potential.potential, np.ndarray)
    assert len(potential.potential.shape) == 1
    if not hasattr(potential, 'dist'):
        raise AssertionError('The potential must possess a field "d" ' +
                             'corresponding to the physical distance between ' +
                             'points on the potential grid.')
    assert potential.dist > 0
    assert isinstance(k, int)
    assert k >= 1

    v = potential.potential
    d = potential.dist

    n_x = v.shape[0]

    d2 = -2*np.diag(np.ones((1, n_x))[0]) \
        + 1*np.diag(np.ones((1, n_x-1))[0], k=-1) \
        + 1*np.diag(np.ones((1, n_x-1))[0], k=1)

    d2 = d2 / d**2

    # Create the Hamiltonian for a single electron
    kinetic = (-hbar**2/(2*me*Mass)) * d2
    potential = np.diag(v)
    hamiltonian = kinetic + potential

    # Reconstruct the physical coordinates of the grid. This doesn't have to
    # agree with the linspace used in generating the potential since only
    # relative distances matter.
    x = np.linspace(0, n_x-1, num=n_x)*d

    # Calculate the interaction term and set v_ee = np.max(v) for diffs == 0
    rdiff = abs(np.diff(cartesian((x, x))))[:, 0]
    rdiff[rdiff == 0] = 1 / np.max(v)
    v_ee = np.diag(1 / rdiff)

    # Calculate the full Hamiltonian: H = h x I + I x h + v_ee
    hamiltonian = np.kron(hamiltonian, np.eye(n_x)) + np.kron(np.eye(n_x), hamiltonian)
    hamiltonian = hamiltonian + v_ee

    energy, psi = eigs(hamiltonian, k=k, which='SM')

    # Drop the imaginary component (which is always 0)
    energy = np.real(energy)

    # Reformat the eigenstates so that they are reformatted as x1 vs x2
    psi = np.array([np.transpose(psi[:, i]).flatten().reshape((n_x, n_x))
                    for i in range(psi.shape[1])])

    return energy, psi


def __schrodinger2d__(potential, k=1):
    """
    Numerically solve the two-dimensional Schrodinger equation with a single
    electron.

    Args:
        **potential (np.ndarray)**: A grid of potential values.
        **k (int)**: The number of eigenvectors and eigenvalues to return.

    Returns:
        **E (numpy.array)**: An array of energy eigenvalues.
        **psi (numpy.array)**: An array of size (k, n_x*n_y) containing the
            amplitude values of the solution to the Schrodinger equation.
    """
    assert isinstance(potential.potential, np.ndarray)
    assert len(potential.potential.shape) == 2
    if not hasattr(potential, 'dist'):
        raise AssertionError('The potential must possess a field "d" ' +
                             'corresponding to the physical distance between ' +
                             'points on the potential grid.')
    assert potential.dist > 0
    assert isinstance(k, int)
    assert k >= 1

    v = potential.potential
    d = potential.dist

    n_x, n_y = v.shape[1], v.shape[0]

    a_xy = np_matlib.repmat(np.ones((1, n_x-1)), n_y, 1).flatten()

    dx2 = -2*np.diag(np.ones((1, n_y*n_x))[0]) \
        + 1*np.diag(a_xy, k=-n_y) \
        + 1*np.diag(a_xy, k=n_y)

    dx2 = dx2 / d**2

    dy22 = -2*np.diag(np.ones((1, n_y))[0]) \
        + 1*np.diag(np.ones((1, n_y-1))[0], -1) \
        + 1*np.diag(np.ones((1, n_y-1))[0], 1)

    dy2 = np.kron(np.eye(n_x), dy22)

    dy2 = dy2 / d**2

    # Create kinetic and potential energy operators and then assign the Hamiltonian
    kin = (-hbar**2/(2*me*Mass)) * (dx2 + dy2)
    pot = np.diag(v.flatten())*e
    ham = kin + pot

    energy, psi = eigs(ham, k=k, which='SM')

    # Drop the imaginary component (which is always 0)
    energy = np.real(energy)

    # Reformat the eigenstates so that they are returned in the shape of the potential
    psi = np.array([np.transpose(psi[:, i]).flatten().reshape((n_x, n_y))
                    for i in range(psi.shape[1])])

    return energy, psi
