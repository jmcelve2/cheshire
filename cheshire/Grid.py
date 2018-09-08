import numpy as np


class Grid1D(object):
    """
    The physical grid for 1D Potentials.

        **Attributes:**
            **n_x (int)**: The number of grid points along the x axis. This number must be a multiple of 2.

            **n_e (int)**: The number of electrons on the grid.

            **x_min (float)**: The minimum physical distance (in a.u.) on the potential grid along the x axis.

            **x_max (float)**: The maximum physical distance (in a.u.) on the potential grid along the x axis.

            **x (numpy.array)**: A grid of distances (in a.u.) from the center of the grid along the x axis.
    """

    def __init__(self, n_x=128, n_e=2, x_min=-20, x_max=20):
        """
        Grid1D constructor.

        **Args:**
            **n_x (int)**: The number of grid points along the x axis. This number must be a multiple of 2. Default is
            128.

            **n_e (int)**: The number of electrons on the grid.

            **x_min (float)**: The minimum physical distance (in a.u.) on the potential grid along the x axis. Default
            is -20.

            **x_max (float)**: The maximum physical distance (in a.u.) on the potential grid along the x axis. Default
            is 20.
        """

        if not ((n_x & (n_x - 1)) == 0) and n_x != 0:
            raise AssertionError('n_x must be must be an integer power of 2.')
        assert n_e == 2
        assert x_min < x_max

        self.n_x = n_x
        self.n_e = n_e
        self.x_min = x_min
        self.x_max = x_max
        self.x = np.linspace(start=x_min, stop=x_max, num=n_x)
        self.params = dict(n_x=n_x, n_e=n_e, x_min=x_min, x_max=x_max, x=self.x)

    def rescale_by_size(self, val):
        """
        Rescale parameters by the pixel size of the grid.
        """
        return val * self.x_max / 20

    def rescale_by_grid(self, val):
        """
        Rescale parameters by the physical size of the grid.
        """
        return val * self.n_x / 256


class Grid2D(object):
    """
    The physical grid for 2D Potentials.

        **Attributes:**
            **n_x (int)**: The number of grid points along the x axis. This number must be a multiple of 2.

            **n_y (int)**: The number of grid points along the y axis. This number must be a multiple of 2.

            **n_e (int)**: The number of electrons on the grid.

            **x_min (float)**: The minimum physical distance (in a.u.) on the potential grid along the x axis.

            **x_max (float)**: The maximum physical distance (in a.u.) on the potential grid along the x axis.

            **y_min (float)**: The minimum physical distance (in a.u.) on the potential grid along the y axis.

            **y_max (float)**: The maximum physical distance (in a.u.) on the potential grid along the y axis.

            **x (numpy.array)**: A grid of distances (in a.u.) from the center of the grid along the x axis.

            **y (numpy.array)**: A grid of distances (in a.u.) from the center of the grid along the y axis.
    """

    def __init__(self, n_x=128, n_y=128, n_e=1, x_min=-20, x_max=20, y_min=-20, y_max=20):
        """
        Grid2D constructor.

        **Args:**
            **n_x (int)**: The number of grid points along the x axis. This number must be a multiple of 2. Default is
            128.

            **n_y (int)**: The number of grid points along the y axis. This number must be a multiple of 2. Default is
            128.

            **n_e (int)**: The number of electrons on the grid.

            **x_min (float)**: The minimum physical distance (in a.u.) on the potential grid along the x axis. Default
            is -20.

            **x_max (float)**: The maximum physical distance (in a.u.) on the potential grid along the x axis. Default
            is 20.

            **y_min (float)**: The minimum physical distance (in a.u.) on the potential grid along the y axis. Default
            is -20.

            **y_max (float)**: The maximum physical distance (in a.u.) on the potential grid along the y axis. Default
                is 20.
        """

        if not ((n_x & (n_x - 1)) == 0) and n_x != 0:
            raise AssertionError('n_x must be must be an integer power of 2.')
        if not ((n_y & (n_y - 1)) == 0) and n_y != 0:
            raise AssertionError('n_y must be must be an integer power of 2.')
        assert n_e == 1
        assert n_x == n_y
        assert x_min < x_max
        assert y_min < y_max
        assert (x_max - x_min) == (y_max - y_min)

        self.n_x = n_x
        self.n_y = n_y
        self.n_e = n_e
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x = np.tile(np.linspace(start=x_min, stop=x_max, num=n_x), (n_y, 1))
        self.y = -np.transpose(np.tile(np.linspace(start=y_min, stop=y_max, num=n_y), (n_x, 1)))
        self.params = dict(n_x=n_x, n_y=n_y, n_e=n_e, x_min=x_min, x_max=x_max, x=self.x, y_min=y_min, y_max=y_max,
                           y=self.y)

    def rescale_by_size(self, val):
        """
        Rescale parameters by the pixel size of the grid.
        """
        return val * self.x_max / 20

    def rescale_by_grid(self, val):
        """
        Rescale parameters by the physical size of the grid.
        """
        return val * self.n_x / 256
