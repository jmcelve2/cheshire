
import numpy as np
import numpy.matlib as np_matlib
from cheshire.Grid import *

HBAR = 1
MASS_E = 1
MASS = 1


class Kinetic(object):
    """
    A kinetic object containing information about the kinetic energy.

    **Attributes:**
        **kinetic (numpy.array):** A grid of values corresponding to the kinetic energy.
    """

    def __init__(self, params):
        """
        A kinetic object containing information about the kinetic energy
        operator.

            **Args:**
                **params (dict):** A dictionary of values specifying the type of kinetic energy operator to create.
        """
        if "kinetic" not in params.keys():
            raise AssertionError("The params dictionary must contain a kinetic grid.")
        assert isinstance(params["kinetic"], np.ndarray)

        for key in params:
            setattr(self, key, params[key])


class Kinetic1D(Grid1D):
    """
    One dimensional kinetic energy operator.
    """

    def create(self):
        """
        Add the "kinetic" attribute to the Kinetic1D object.

            **Args:**
                **c_x (float)**: The center of the infinite well along the x axis (in a.u.). Default is 0.

                **l_x (float)**: The length of the infinite well along the x axis (in a.u.). Default is 7.

            **Returns:**
                A numpy.array grid of potential values with units of Hartree
                energy.
        """
        d = abs(self.x[1]-self.x[0])

        d2 = -2*np.diag(np.ones((1, self.n_x))[0]) \
             + 1*np.diag(np.ones((1, self.n_x-1))[0], k=-1) \
             + 1*np.diag(np.ones((1, self.n_x-1))[0], k=1)

        d2 = d2 / d**2

        # Create the Hamiltonian for a single electron
        kinetic = (-HBAR**2/(2*MASS_E*MASS)) * d2

        return Kinetic(dict(**self.params, **dict(kinetic=kinetic)))


class Kinetic2D(Grid2D):
    """
    Two dimensional kinetic energy operator.
    """

    def create(self):
        """
        Add the "kinetic" attribute to the Kinetic2D object.

            **Returns:**
                A numpy.array grid of kinetic values with units of Hartree
                energy.
        """
        d = abs(self.x[0][1]-self.x[0][0])

        a_xy = np_matlib.repmat(np.ones((1, self.n_x-1)), self.n_y, 1).flatten()

        dx2 = -2*np.diag(np.ones((1, self.n_y*self.n_x))[0]) \
              + 1*np.diag(a_xy, k=-self.n_y) \
              + 1*np.diag(a_xy, k=self.n_y)

        dx2 = dx2 / d**2

        dy22 = -2*np.diag(np.ones((1, self.n_y))[0]) \
               + 1*np.diag(np.ones((1, self.n_y-1))[0], -1) \
               + 1*np.diag(np.ones((1, self.n_y-1))[0], 1)

        dy2 = np.kron(np.eye(self.n_x), dy22)

        dy2 = dy2 / d**2

        kinetic = (-HBAR**2/(2*MASS_E*MASS)) * (dx2 + dy2)

        return Kinetic(dict(**self.params, **dict(kinetic=kinetic)))


def kinetic_factory(grid_params):
    """
    Infer the Kinetic to create based on the parameters passed to this method.
    """
    assert isinstance(grid_params, dict)

    if 'n_x' in grid_params.keys() and 'n_y' not in grid_params.keys():
        kinetic = Kinetic1D(**grid_params)
    elif 'n_x' in grid_params.keys() and 'n_y' in grid_params.keys():
        kinetic = Kinetic2D(**grid_params)
    else:
        raise AssertionError('The parameters passed into the creation method are not supported.')

    return kinetic.create()
