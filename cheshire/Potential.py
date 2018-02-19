# coding=utf-8
import numpy as np
from scipy import ndimage
from scipy.interpolate import spline
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from cheshire.Grid import *


class Potential(object):
    """
    A potential object containing information about the potential.
    """

    def __init__(self, params):
        """
        A potential object containing information about the potential.

        Args:
            **params (dict)**: A dictionary of values specifying the type of potential to create.
        """
        if not "potential" in params.keys():
            raise AssertionError("The params dictionary must contain a potential grid.")
        assert isinstance(params["potential"], np.ndarray)

        for key in params:
            setattr(self, key, params[key])


class InfiniteWell1D(Grid1D):
    """
    One dimensional infinite well potential.
    """

    def create(self, c_x=0, l_x=7):
        """
        Generate a Potential object with a 1D infinite well potential.

        Args:
            **c_x (float)**: The center of the infinite well along the x axis (in a.u.). Default is 0.
            **l_x (float)**: The length of the infinite well along the x axis (in a.u.). Default is 7.

        Returns:
            An object of class Potential with a potential attribute and attributes corresponding to the parameters used
            to generate the potential.
        """

        assert self.x_min < c_x-0.5*l_x
        assert c_x+0.5*l_x < self.x_max

        v = np.zeros(shape=(1, self.n_x))[0]
        mask = (c_x-0.5*l_x < self.x) & (self.x <= c_x+0.5*l_x)

        v[~mask] = 20

        params = dict(**self.params, **dict(potential=v, c_x=c_x, l_x=l_x))

        return Potential(params=params)


class Coulomb1D(Grid1D):
    """
    One dimensional Coulomb potential.
    """

    def create(self, c_x=0, z=1):
        """
        Generate a Potential object with a 1D Coulomb potential.

        Args:
            **c_x (float)**: The center of the Coulomb potential along the x axis (in a.u.). Default is 0.
            **z (int)**: The number of protons determining the strength of the Coulomb attraction. Default is 1.

        Returns:
            An object of class Potential with a potential attribute and attributes corresponding to the parameters used
            to generate the potential.
        """
        assert isinstance(z, int)
        assert z >= 1
        assert (c_x > self.x_min) and (c_x < self.x_max)

        v = z/np.sqrt((c_x-self.x)**2+alpha**2)
        v = np.max(v)-v

        params = dict(**self.params, **dict(potential=v, z=z, c_x=c_x, alpha=alpha))

        return Potential(params=params)


class InfiniteWell2D(Grid2D):
    """
    Two dimensional infinite well potential.
    """

    def create(self, c_x=0, c_y=0, l_x=7, l_y=7):
        """
        Generate a Potential object with a 2D infinite well potential.

        Args:
            **c_x (float)**: The center of the infinite well along the x axis (in a.u.). Default is 0.
            **c_y (float)**: The center of the infinite well along the y axis (in a.u.). Default is 0.
            **l_x (float)**: The length of the infinite well along the x axis (in a.u.). Default is 7.
            **l_y (float)**: The length of the infinite well along the y axis (in a.u.). Default is 7.

        Returns:
            An object of class Potential with a potential attribute and attributes corresponding to the parameters used
            to generate the potential.
        """

        assert (c_x > self.x_min) & (c_x < self.x_max)
        assert (c_y > self.y_min) & (c_y < self.y_max)
        assert (l_x > self.x_min) & (l_x < self.x_max)
        assert (l_y > self.y_min) & (l_y < self.y_max)

        v = np.zeros(shape=(self.n_y, self.n_x))
        mask = (c_x-0.5*l_x < self.x) & (self.x <= c_x+0.5*l_x) & \
               (c_y-0.5*l_y < self.y) & (self.y <= c_y+0.5*l_y)

        v[~mask] = 20

        params = dict(**self.params, **dict(potential=v, c_x=c_x, c_y=c_y, l_x=l_x, l_y=l_y))

        return Potential(params=params)


class SimpleHarmonicOscillator2D(Grid2D):
    """
    Two dimensional simple harmonic oscillator potential.
    """

    def create(self, c_x=0, c_y=0, k_x=2, k_y=2):
        """
        Generate a Potential object with a 2D simple harmonic oscillator potential.

        Args:
            **c_x (float)**: The center of the simple harmonic oscillator along the x axis (in a.u.). Default is 0.
            **c_y (float)**: The center of the simple harmonic oscillator along the y axis (in a.u.). Default is 0.
            **k_x (float)**: The constant that determines the width of the harmonic oscillator potential along the x
                axis. Default is 2.
            **k_y (float)**: The constant that determines the width of the harmonic oscillator potential along the y
                axis. Default is 2.

        Returns:
            An object of class Potential with a potential attribute and attributes corresponding to the parameters used
            to generate the potential.
        """
        assert (c_x > self.x_min) & (c_x < self.x_max)
        assert (c_y > self.y_min) & (c_y < self.y_max)
        assert k_x > 0
        assert k_y > 0

        v = .5*(k_x*(self.x-c_x)**2+k_y*(self.y-c_y)**2)
        v[v > 20] = 20

        params = dict(**self.params, **dict(potential=v, c_x=c_x, c_y=c_y, k_x=k_x, k_y=k_y))

        return Potential(params=params)


class DoubleInvertedGaussian2D(Grid2D):
    """
    Two dimensional infinite well potential.
    """

    def create(self, a1=2, a2=2, c_x1=-2, c_x2=2, c_y1=-2, c_y2=2, k_x1=2, k_x2=2, k_y1=2, k_y2=2):
        """
        Generate a Potential object with a 2D double inverted Gaussian potential.

        Args:
            **a1 (float)**: The maximum depth of the first Gaussian in units of Hartree Energy. Default is 2.
            **a2 (float)**: The maximum depth of the first Gaussian in units of Hartree Energy. Default is 2.
            **c_x1 (float)**: The center of the first Gaussian well along the x axis (in a.u.). Default is -2.
            **c_y1 (float)**: The center of the first Gaussian well along the y axis (in a.u.). Default is 2.
            **c_x2 (float)**: The center of the second Gaussian well along the x axis (in a.u.). Default is -2.
            **c_y2 (float)**: The center of the second Gaussian well along the y axis (in a.u.). Default is 2.
            **k_x1 (float)**: The constant that determines the width of the first inverted Gaussian along the x axis.
                Default is 2.
            **k_x2 (float)**: The constant that determines the width of the second inverted Gaussian along the x axis.
                Default is 2.
            **k_y1 (float)**: The constant that determines the width of the first inverted Gaussian along the y axis.
                Default is 2.
            **k_y2 (float)**: The constant that determines the width of the second inverted Gaussian along the y axis.
                Default is 2.

        Returns:
            An object of class Potential with a potential attribute and attributes corresponding to the parameters used
            to generate the potential.
        """

        assert a1 > 0
        assert a2 > 0
        assert (c_x1 > self.x_min) & (c_x1 < self.x_max)
        assert (c_x2 > self.x_min) & (c_x2 < self.x_max)
        assert (c_y1 > self.y_min) & (c_y1 < self.y_max)
        assert (c_y2 > self.y_min) & (c_y2 < self.y_max)
        assert k_x1 > 0
        assert k_x2 > 0
        assert k_y1 > 0
        assert k_y2 > 0

        v = -a1*np.exp(-((self.x-c_x1)/k_x1)**2 - ((self.y-c_y1)/k_y1)**2) \
            - a2*np.exp(-((self.x-c_x2)/k_x2)**2 - ((self.y-c_y2)/k_y2)**2)

        v = v + np.max(v)

        params = dict(**self.params, **dict(potential=v, a1=a1, a2=a2, c_x1=c_x1, c_x2=c_x2, c_y1=c_y1, c_y2=c_y2,
                                            k_x1=k_x1, k_x2=k_x2, k_y1=k_y1, k_y2=k_y2))

        return Potential(params=params)


class Random2D(Grid2D):
    """
    Two dimensional random potential.

    To generate a random potential, a 16 x 16 binary grid of 1s and 0s is generated and upscaled to n x n. A second
    16 x 16 binary grid is generated and upscaled to n/2 x n/2. The smaller grid is centered within the larger grid and
    then the grids are subtracted element-wise. A Gaussian blur is then applied with standard deviation sig1**2. The
    potential is now random, and smooth, but does not achieve a maximum at the boundary.

    To achieve this, a mask that smoothly goes to zero at the boundary and 1 in the interior is generated. To generate
    the desired random mask, k**2 random coordinate pairs are generated on a 200*n/256 x 200*n/256 grid. A convex hull
    is generated with these points, and the boundary of the convex hull is smoothly interpolated using a cubic spline. A
    binary mask is then formed by filling the inside of the closed blob with 1s, and the outside with 0s. Resizing the
    blob to a resolution of r x r, and applying a Gaussian blur with standard deviation sig2 returns the final mask.

    Element-wise multiplication of the mask with the random-blurred image gives a random potential that approaches zero
    at the boundary. The “sharpness” of the potential is randomized by then exponentiating by either d = 0.1, 0.5, 1.0,
    or 2.0, chosen at random with equal probabilities (i.e. V := V**p). The result is then subtracted from its maximum
    to invert the well.
    """

    def create(self, k=5, r=40, sig1=8, sig2=13, p=2):
        """
        Generate a Potential object with a 2D random potential.

        Args:
            **k (int)**: Determines the number of integers (k**2) to use to generate the convex hull that makes the blob.
            **r (float)**: The resolution size of the blob.
            **sig1 (float)**: The variance of the first Gaussian blur.
            **sig2 (float)**: The variance of the second Gaussian blur.
            **p (float)**: The exponent used to increase the contrast of the potential.

        Returns:
            An object of class Potential with a potential attribute and attributes corresponding to the parameters used
            to generate the potential.
        """
        assert (k >= 2) & (k <= 7)
        assert (self.rescale_by_grid(r) < self.n_x) & (r > 0)
        assert sig1 > 0
        assert sig2 > 0
        assert p in [0.5, 1, 1.5, 2]

        # Create the n x n and n/2 x n/2 grids
        v = np.reshape(np.random.randint(low=0, high=2, size=16*16), newshape=(16,16))
        v = np.kron(v, np.ones((round(self.n_x/16), round(self.n_y/16))))
        subgrid = np.reshape(np.random.randint(low=0, high=2, size=16*16), newshape=(16,16))
        subgrid = np.kron(subgrid, np.ones((round(self.n_x/32), round(self.n_y/32))))

        # Center and diff the two grids
        lo_x = round(self.n_x/4)
        hi_x = round(self.n_x/4*3)
        lo_y = round(self.n_y/4)
        hi_y = round(self.n_y/4*3)
        v[lo_y:hi_y, lo_x:hi_x] = v[lo_y:hi_y, lo_x:hi_x] - subgrid

        # Run the first Gaussian blur
        v = gaussian_filter(v, sigma=sig1)

        # Create the convex hull from k**2 points
        points = np.random.rand(k**2, 2)*(self.rescale_by_grid(200))
        points = points.astype(int)
        hull = ConvexHull(points)

        # Get the x and y points of the convex hull
        x = np.transpose(hull.points[hull.vertices])[0]
        x = np.append(x, x[0])
        y = np.transpose(hull.points[hull.vertices])[1]
        y = np.append(y, y[0])

        # Parameterize the boundary of the hull and interpolate
        t = np.arange(x.shape[0], dtype=float)
        t /= t[-1]
        nt = np.linspace(0, 1, 100)
        x = spline(t, x, nt)
        y = spline(t, y, nt)

        # Create a Delaunay hull for identifying points inside of the hull
        # The two args of np.zeros need to be scaled independently when n_x and n_y are allowed to vary
        # independently.
        hull = ConvexHull(np.transpose(np.array((x, y))))
        hull = Delaunay(hull.points[hull.vertices])
        coords = np.transpose(np.indices((round(self.rescale_by_grid(200)), round(self.rescale_by_grid(200)))))
        coords = coords.reshape(round(self.rescale_by_grid(200)**2), 2)
        in_hull = hull.find_simplex(coords) >= 0

        # Rescale the blob
        # The two args of np.zeros need to be scaled independently when n_x and n_y are allowed to vary
        # independently.
        blob = np.zeros(shape=(round(self.rescale_by_grid(200)), round(self.rescale_by_grid(200))))
        blob[np.transpose(coords[in_hull])[0], np.transpose(coords[in_hull])[1]] = 1
        blob = ndimage.zoom(blob, self.rescale_by_grid(r)/self.rescale_by_grid(200))

        # Create the final mask with the second Gaussian blur
        mask = np.zeros(shape=v.shape)
        x_offset = round((mask.shape[0]-blob.shape[0])/2)
        y_offset = round((mask.shape[1]-blob.shape[1])/2)
        mask[x_offset:blob.shape[0]+x_offset, y_offset:blob.shape[1]+y_offset] = blob
        mask = gaussian_filter(mask, sigma=sig2)

        v = np.abs(v)
        v = v**p
        v = v*mask
        v = np.max(v) - v + (1 - np.max(v))

        params = dict(**self.params, **dict(potential=v, k=k, r=r, sig1=sig1, sig2=sig2, p=p))

        return Potential(params=params)


class Coulomb2D(Grid2D):
    """
    Two dimensional Coulomb potential.
    """

    def create(self, z=1, c_x=0, c_y=0, alpha=10**-9):
        """
        Generate a Potential object with a 2D Coulomb potential.

        Args:
            **z (int)**: Determines the magnitude of the potential. An effective "proton number" constant.
            **c_x (float)**: The center of the Coulomb potential along the x axis (in a.u.). Default is 0.
            **c_y (float)**: The center of the Coulomb potential along the y axis (in a.u.). Default is 0.
            **alpha (float)**: A value that removes the Coulomb singularity to ensure the solver converges.

        Returns:
            An object of class Potential with a potential attribute and attributes corresponding to the parameters used
            to generate the potential.
        """

        assert isinstance(z, int)
        assert z >= 1
        assert (c_x > self.x_min) and (c_x < self.x_max)
        assert (c_y > self.y_min) and (c_y < self.y_max)

        v = z/np.sqrt((c_x-self.x)**2+(c_y-self.y)**2+alpha**2)
        v = v.max()-v

        params = dict(**self.params, **dict(potential=v, z=z, c_x=c_x, c_y=c_y, alpha=alpha))

        return Potential(params=params)


def potential_factory(grid_params, pot_params):
    """
    Infer the Potential to create based on the parameters passed to this method.
    """
    assert isinstance(grid_params, dict)
    assert isinstance(pot_params, dict)

    if 'n_x' in grid_params.keys() and 'n_y' not in grid_params.keys():
        pass
    elif 'n_x' in grid_params.keys() and 'n_y' in grid_params.keys():
        pass
    else:
        raise AssertionError('The parameters passed into the creation method are not supported.')

    if set(pot_params) == set(['c_x', 'l_x']):
        potential = InfiniteWell1D(**grid_params)
    elif set(pot_params) == set(['c_x', 'c_y', 'l_x', 'l_y']):
        potential = InfiniteWell2D(**grid_params)
    elif set(pot_params) == set(['c_x', 'c_y', 'k_x', 'k_y']):
        potential = SimpleHarmonicOscillator2D(**grid_params)
    elif set(pot_params) == set(['a1', 'a2', 'c_x1', 'c_x2', 'c_y1', 'c_y2', 'k_x1', 'k_y1', 'k_x2', 'k_y2']):
        potential = DoubleInvertedGaussian2D(**grid_params)
    elif set(pot_params) == set(['k', 'r', 'sig1', 'sig2', 'p']):
        potential = Random2D(**grid_params)
    elif set(pot_params) == set(['z', 'c_x', 'c_y', 'alpha']):
        potential = Coulomb2D(**grid_params)
    else:
        raise AssertionError('None of the supported potentials accept the parameters passed into the creation method.')

    return potential.create(**pot_params)
