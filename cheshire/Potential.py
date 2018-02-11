
import numpy as np
from scipy import ndimage
from scipy.interpolate import spline
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay


class Potential:
    """
    A potential object containing information about the potential.

    Attributes:
        **potential (numpy.array)**: A grid of values corresponding to the
            potential energy.
        **dist (float)**: The physical distance between adjacent points
            on a grid. This is one number because the grid is assumed to
            be homogeneous.
    """

    def __init__(self, potential, dist):
        """
        A potential object containing information about the potential.

        Args:
            **potential (numpy.array)**: A grid of values corresponding to the
                potential energy.
            **dist (float)**: The physical distance between adjacent points
                on a grid. This is one number because the grid is assumed to
                be homogeneous.
        """

        assert isinstance(potential, np.ndarray)
        assert isinstance(dist, float)

        self.potential = potential
        self.dist = dist


class PotentialFactory:
    """
    The potential factory class used to instantiate instances of potentials.

    Attributes:
        **n_x (int)**: The number of grid points along the x axis. This number
            must be a multiple of 2.
        **n_y (int)**: The number of grid points along the y axis. This number
            must be a multiple of 2.
        **x_min (float)**: The minimum physical distance (in a.u.) on the
            potential grid along the x axis.
        **x_max (float)**: The maximum physical distance (in a.u.) on the
            potential grid along the x axis.
        **y_min (float)**: The minimum physical distance (in a.u.) on the
            potential grid along the y axis.
        **y_max (float)**: The maximum physical distance (in a.u.) on the
            potential grid along the y axis.
        **x (numpy.array)**: A grid of distances (in a.u.) from the center 
            of the grid along the x axis.
        **y (numpy.array)**: A grid of distances (in a.u.) from the center 
            of the grid along the y axis.
        **d (float)**: The distance (in a.u.) between neighboring grid points.
    """
    
    def __init__(self, n_x=128, n_y=128, 
                 x_min=-20, x_max=20, y_min=-20, y_max=20):
        """
        Potential constructor.

        Args:
            **n_x (int)**: The number of grid points along the x axis. This number
                must be a multiple of 2. Default is 128.
            **n_y (int)**: The number of grid points along the y axis. This number
                must be a multiple of 2. Default is 128.
            **x_min (float)**: The minimum physical distance (in a.u.) on the
                potential grid along the x axis. Default is -20.
            **x_max (float)**: The maximum physical distance (in a.u.) on the
                potential grid along the x axis. Default is 20.
            **y_min (float)**: The minimum physical distance (in a.u.) on the
                potential grid along the y axis. Default is -20.
            **y_max (float)**: The maximum physical distance (in a.u.) on the
                potential grid along the y axis. Default is 20.
        """

        if not ((n_x & (n_x - 1)) == 0) and n_x != 0:
            raise AssertionError('n_x must be must be an integer power of 2.')
        if not ((n_y & (n_y - 1)) == 0) and n_y != 0:
            raise AssertionError('n_y must be must be an integer power of 2.')
        assert n_x == n_y
        assert x_min < x_max
        assert y_min < y_max
        assert (x_max - x_min) == (y_max - y_min)

        self.n_x = n_x
        self.n_y = n_y
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x = np.tile(np.linspace(start=x_min,
                                     stop=x_max,
                                     num=n_x),
                         (n_y, 1))
        self.y = -np.transpose(np.tile(np.linspace(start=y_min,
                                                   stop=y_max,
                                                   num=n_y),
                                       (n_x, 1)))
        self.d = abs(self.x[0][0]-self.x[0][1])
        
    def iw(self, c_x=0, c_y=0, l_x=7, l_y=7):
        """
        Generate an approximate infinite well potential.

        Args:
            **c_x (float)**: The center of the infinite well along the x
                axis (in a.u.). Default is 0.
            **c_y (float)**: The center of the infinite well along the y
                axis (in a.u.). Default is 0.
            **l_x (float)**: The length of the infinite well along the x
                axis (in a.u.). Default is 7.
            **l_y (float)**: The length of the infinite well along the y
                axis (in a.u.). Default is 7.

        Returns:
            A numpy.array grid of potential values with units of Hartree 
            energy.
        """
        
        assert (c_x > self.x_min) & (c_x < self.x_max)
        assert (c_y > self.y_min) & (c_y < self.y_max)
        assert (l_x > self.x_min) & (l_x < self.x_max)
        assert (l_y > self.y_min) & (l_y < self.y_max)

        v = np.zeros(shape=(self.n_y, self.n_x))
        mask = (.5*(2*c_x-l_x) < self.x) & (self.x <= .5*(2*c_x+l_x)) & \
            (.5*(2*c_y-l_y) < self.y) & (self.y <= .5*(2*c_y+l_y))
            
        v[~mask] = 20
        
        return Potential(potential=v, dist=self.d)
        
    def sho(self, c_x=0, c_y=0, k_x=2, k_y=2):
        """
        Generate an approximate simple harmonic oscillator potential.

        Args:
            **c_x (float)**: The center of the simple harmonic oscillator 
                along the x axis (in a.u.). Default is 0.
            **c_y (float)**: The center of the simple harmonic oscillator 
                along the y axis (in a.u.). Default is 0.
            **k_x (float)**: The constant that determines the width of the 
                harmonic oscillator potential along the x axis. Default is 2.
            **k_y (float)**: The constant that determines the width of the 
                harmonic oscillator potential along the y axis. Default is 2.

        Returns:
            A numpy.array grid of potential values with units of Hartree 
            energy.
        """
        
        v = .5*(k_x*(self.x-c_x)**2+k_y*(self.y-c_y)**2)
        v[v > 20] = 20

        return Potential(potential=v, dist=self.d)
        
    def dig(self, a1=2, a2=2, 
            c_x1=-2, c_x2=2, c_y1=-2, c_y2=2, 
            k_x1=2, k_y1=2, k_x2=2, k_y2=2):
        """
        Generate a double inverted Gaussian potential.

        Args:
            **a1 (float)**: The maximum depth of the first Gaussian in
                units of Hartree Energy. Default is 2.
            **a2 (float)**: The maximum depth of the first Gaussian in
                units of Hartree Energy. Default is 2.
            **c_x1 (float)**: The center of the first Gaussian well along 
                the x axis (in a.u.). Default is -2.
            **c_y1 (float)**: The center of the first Gaussian well along 
                the y axis (in a.u.). Default is 2.
            **c_x2 (float)**: The center of the second Gaussian well along 
                the x axis (in a.u.). Default is -2.
            **c_y2 (float)**: The center of the second Gaussian well along 
                the y axis (in a.u.). Default is 2.
            **k_x1 (float)**: The constant that determines the width of the 
                first inverted Gaussian along the x axis. Default is 2.
            **k_y1 (float)**: The constant that determines the width of the 
                first inverted Gaussian along the y axis. Default is 2.
            **k_x2 (float)**: The constant that determines the width of the 
                second inverted Gaussian along the x axis. Default is 2.
            **k_y2 (float)**: The constant that determines the width of the 
                second inverted Gaussian along the y axis. Default is 2.

        Returns:
            A numpy.array grid of potential values with units of Hartree 
            energy.
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

        v = v + np.min(v)

        return Potential(potential=v, dist=self.d)
        
    def rand(self, k=5, r=40, sig1=8, sig2=13, p=2):
        """
        Generate a random potential. 

        To generate a random potential, a 16 x 16 binary grid of 1s and
        0s is generated and upscaled to n x n. A second 16 x 16 binary
        grid is generated and upscaled to n/2 x n/2. The smaller grid is
        centered within the larger grid and then the grids are subtracted 
        element-wise. A Gaussian blur is then applied with standard deviation
        sig1**2. The potential is now random, and smooth, but does not achieve 
        a maximum at the boundary.

        To achieve this, a mask that smoothly goes to zero at the 
        boundary and 1 in the interior is generated. To generate 
        the desired random mask, k**2 random coordinate pairs are 
        generated on a 200*n/256 x 200*n/256 grid. A convex hull is
        generated with these points, and the boundary of the convex hull is
        smoothly interpolated using a cubic spline. A binary mask 
        is then formed by filling the inside of the closed blob with 
        1s, and the outside with 0s. Resizing the blob to a resolution
        of r x r, and applying a Gaussian blur with standard deviation
        sig2 returns the final mask.

        Element-wise multiplication of the mask with the random-blurred
        image gives a random potential that approaches zero at the 
        boundary. The “sharpness” of the potential is randomized by 
        then exponentiating by either d = 0.1, 0.5, 1.0, or 2.0, chosen 
        at random with equal probabilities (i.e. V := V**p). The result
        is then subtracted from its maximum to invert the well.

        Args:
            **k (int)**: Determines the number of integers (k**2) to use
                to generate the convex hull that makes the blob.
            **r (float)**: The resolution size of the blob.
            **sig1 (float)**: The variance of the first Gaussian blur.
            **sig2 (float)**: The variance of the second Gaussian blur.
            **p (float)**: The exponent used to increase the contrast of
                the potential.

        Returns:
            A numpy.array grid of potential values with units of Hartree 
            energy.
        """

        assert (k >= 2) & (k <= 7)
        assert (r < self.n_x) & (r > 0)
        assert sig1 > 0
        assert sig2 > 0
        assert isinstance(p, float)
        assert p > 0

        # Create the n x n and n/2 x n/2 grids
        v = np.reshape(np.random.randint(low=0, 
                                         high=2, 
                                         size=16*16), 
                       newshape=(16,16))
        
        v = np.kron(v, np.ones((round(self.n_x/16), 
                                round(self.n_y/16))))

        subgrid = np.reshape(np.random.randint(low=0, 
                                               high=2, 
                                               size=16*16), 
                             newshape=(16,16))
        
        subgrid = np.kron(subgrid, np.ones((round(self.n_x/32), 
                                            round(self.n_y/32))))
        
        lo_x = round(self.n_x/4)
        hi_x = round(self.n_x/4*3)
        lo_y = round(self.n_y/4)
        hi_y = round(self.n_y/4*3)

        # Center and diff the two grids
        v[lo_y:hi_y, lo_x:hi_x] = v[lo_y:hi_y, lo_x:hi_x] - subgrid

        # Run the first Gaussian blur
        v = gaussian_filter(v, sigma=sig1)

        # Create the convex hull from k**2 points
        points = np.random.rand(k**2, 2)*(200*self.n_x/256)
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
        hull = ConvexHull(np.transpose(np.array((x, y))))
        hull = Delaunay(hull.points[hull.vertices])
        coords = np.transpose(np.indices((round(200*self.n_x/256), 
                                          round(200*self.n_x/256))))
        coords = coords.reshape(round(200*self.n_x/256*200*self.n_x/256), 2)
        in_hull = hull.find_simplex(coords) >= 0

        # Rescale the blob
        blob = np.zeros(shape=(round(200*self.n_x/256), 
                               round(200*self.n_y/256)))
        blob[np.transpose(coords[in_hull])[0], 
             np.transpose(coords[in_hull])[1]] = 1
        blob = ndimage.zoom(blob, (r*self.n_x/256)/(200*self.n_x/256))

        # Create the final mask with the second Gaussian blur
        mask = np.zeros(shape=v.shape)
        x_offset = round((mask.shape[0]-blob.shape[0])/2)
        y_offset = round((mask.shape[1]-blob.shape[1])/2)
        mask[x_offset:blob.shape[0]+x_offset, 
             y_offset:blob.shape[1]+y_offset] = blob
        mask = gaussian_filter(mask, sigma=sig2)

        v = np.abs(v)
        v = v**p
        v = v*mask
        v = np.max(v) - v + (1 - np.max(v))

        return Potential(potential=v, dist=self.d)
    
    def coulomb(self, z=1):
        """
        Generate a double inverted Gaussian potential.

        Args:
            **z (int)**: Determines the magnitude of the potential. An
                effective "proton number" constant.

        Returns:
            A numpy.array grid of potential values with units of Hartree 
            energy.
        """

        assert isinstance(z, int)
        assert z >= 1
        
        v = z/np.sqrt(self.x**2+self.y**2)
        v = v.max()-v

        return Potential(potential=v, dist=self.d)
