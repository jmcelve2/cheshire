
import numpy as np
import numpy.matlib as np_matlib
from scipy.sparse.linalg import eigs

# Hartree atomic units are used, so all of these are unity.
hbar = 1
e = 1
me = 1
Mass = 1


def schrodinger(potential, k=1):
    """
    Numerically solve the Schrodinger equation.

    Args:
        **potential (Potential)**: An object of class Potential containing
            an array of potential energy values and distance information.
        **k (int)**: The number of eigenvalues and eigenvectors to return
            from the diagonalization process.

    Yields:
        **E (numpy.array)**: An array of energy eigenvalues.
        **psi (numpy.array)**: An array of size (k, n_x*n_y) containing the
            amplitude values of the solution to the Schrodinger equation.
    """
    assert isinstance(potential.potential, np.ndarray)
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
        
    dx2 = dx2/(d**2)
    
    dy22 = -2*np.diag(np.ones((1, n_y))[0]) \
        + 1*np.diag(np.ones((1, n_y-1))[0], -1) \
        + 1*np.diag(np.ones((1, n_y-1))[0], 1)
        
    dy2 = np.kron(np.eye(n_x), dy22)
    
    dy2 = dy2/(d**2)
    
    hamiltonian = (-hbar**2/(2*me*Mass)) * (dx2 + dy2) + \
        np.diag(v.flatten())*e
    
    energy, psi = eigs(hamiltonian, k=k, which='SM')

    energy = np.real(energy)

    return energy, psi
