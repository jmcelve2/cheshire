
import math
import numpy as np


class ParamSampler:
    """
    The parameter sampler class used to create random parameters for 
    potentials.

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

    """    

    def __init__(self, x_min=-20, x_max=20, y_min=-20, y_max=20,
                 n_x=128, n_y=128):
        """
        Potential constructor.

        Args:
            **n_x (int)**: The number of grid points along the x axis. This number
                must be a multiple of 2.
            **n_y (int)**: The number of grid points along the y axis. This number
                must be a multiple of 2.
            **x_min (float)**: The minimum physical distance (in a.u.) on the
                potential grid along the x axis. Default is -20.
            **x_max (float)**: The maximum physical distance (in a.u.) on the
                potential grid along the x axis. Default is 20.
            **y_min (float)**: The minimum physical distance (in a.u.) on the
                potential grid along the y axis. Default is -20.
            **y_max (float)**: The maximum physical distance (in a.u.) on the
                potential grid along the y axis. Default is 20.
        """

        assert x_min == y_min
        assert x_max == y_max
        assert x_min <= x_max
        assert y_min <= y_max
        assert n_x == n_y

        self.n_x = n_x
        self.n_y = n_y
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def rescale_by_size(self, val, size=20):
        return val*self.x_max/size

    def rescale_by_grid(self, val, grid=256):
        return val*self.n_x/grid
    
    def iw(self, min_l=4, max_l=15, min_c=-8, max_c=8):
        """
        Random parameter method that generates parameters for infinite
        square wells.

        Args:
            **min_l (float)**: The minimum possible allowed length for 
                the well size. Default is 4.
            **max_l (float)**: The maximum possible allowed length for 
                the well size. Default is 15.
            **min_c (float)**: The minimum possible center point for 
                the well. Default is -8.
            **max_c (float)**: The maximum possible center point for 
                the well. Default is 8.

        Returns:
            A dictionary of randomly generated infinite well parameters.
        """
        
        assert min_l <= max_l
        assert min_c <= max_c

        # Rescale arguments based on the size of the grid
        min_l = self.rescale_by_size(min_l)
        max_l = self.rescale_by_size(max_l)
        min_c = self.rescale_by_size(min_c)
        max_c = self.rescale_by_size(max_c)

        # Sample parameters
        def iw_sample():
            
            energy = np.random.uniform(low=0, high=0.4)
            l_x = np.random.uniform(low=min_l*self.x_max/20, high=max_l*self.x_max/20)
            l_y = 1 / np.sqrt((2*energy)/math.pi**2 - 1/l_x**2)
            
            return l_x, l_y
        
        while True:
            l_x, l_y = iw_sample()
            if (min_l <= l_x) & \
                (l_x <= max_l) & \
                (min_l <= l_y) & \
                (l_y <= max_l):
                break
        
        switch = np.random.uniform(low=0, high=1)
        if switch > 0.5:
            temp = l_x
            l_x = l_y
            l_y = temp
        
        l_x = l_x
        l_y = l_y
        c_x = np.random.uniform(low=min_c, high=max_c)
        c_y = np.random.uniform(low=min_c, high=max_c)
        
        return {'l_x': l_x, 'l_y': l_y, 'c_x': c_x, 'c_y': c_y}
    
    def sho(self, min_kx=0, max_kx=0.16, min_ky=0, max_ky=0.16, 
            min_cx=-8, max_cx=8, min_cy=-8, max_cy=8):
        """
        Random parameter method that generates parameters for simple
        harmonic oscillator potentials.

        Args:
            **min_kx (float)**: The minimum possible value for the well
                width along the x axis. Default is 0.
            **max_kx (float)**: The maximum possible value for the well
                width along the x axis. Default is 0.16.
            **min_ky (float)**: The minimum possible value for the well
                width along the y axis. Default is 0.
            **max_ky (float)**: The maximum possible value for the well
                width along the y axis. Default is 0.16.
            **min_cx (float)**: The minimum possible value for the center 
                (in a.u.) of the simple harmonic oscillator along the x 
                axis. Default is -8.
            **max_cx (float)**: The maximum possible value for the center 
                (in a.u.) of the simple harmonic oscillator along the x 
                axis. Default is 8.
            **min_cy (float)**: The minimum possible value for the center 
                (in a.u.) of the simple harmonic oscillator along the y 
                axis. Default is -8.
            **max_cy (float)**: The maximum possible value for the center 
                (in a.u.) of the simple harmonic oscillator along the y 
                axis. Default is 8.

        Returns:
            A dictionary of randomly generated simple harmonic oscillator
            parameters.
        """

        assert min_kx <= max_kx
        assert min_ky <= max_ky
        assert min_cx <= max_cx
        assert min_cy <= max_cy

        # Rescale arguments
        min_kx = self.rescale_by_size(min_kx)
        max_kx = self.rescale_by_size(max_kx)
        min_ky = self.rescale_by_size(min_ky)
        max_ky = self.rescale_by_size(max_ky)
        min_cx = self.rescale_by_size(min_cx)
        max_cx = self.rescale_by_size(max_cx)
        min_cy = self.rescale_by_size(min_cy)
        max_cy = self.rescale_by_size(max_cy)

        # Sample parameters
        k_x = np.random.uniform(low=min_kx, high=max_kx)
        k_y = np.random.uniform(low=min_ky, high=max_ky)
        c_x = np.random.uniform(low=min_cx, high=max_cx)
        c_y = np.random.uniform(low=min_cy, high=max_cy)
        
        return {'k_x': k_x, 'k_y': k_y, 'c_x': c_x, 'c_y': c_y}

    def dig(self, min_a1=2, max_a1=4, min_a2=2, max_a2=4, 
            min_cx1=-8, max_cx1=8, min_cy1=-8, max_cy1=8,
            min_cx2=-8, max_cx2=8, min_cy2=-8, max_cy2=8,
            min_kx1=1.6, max_kx1=8, min_ky1=1.6, max_ky1=8,
            min_kx2=1.6, max_kx2=8, min_ky2=1.6, max_ky2=8):
        """
        Random parameter method that generates parameters for double
        inverted Gaussian potentials.

        Args:
            **min_a1 (float)**: The minimum possible value of the amplitude of
                the first Gaussian. Default is 2. Units are in Hartree
                energy.
            **max_a1 (float)**: The maximum possible value of the amplitude of
                the first Gaussian. Default is 4. Units are in Hartree
                energy.
            **min_a2 (float)**: The minimum possible value of the amplitude of
                the second Gaussian. Default is 2. Units are in Hartree
                energy.
            **max_a2 (float)**: The maximum possible value of the amplitude of
                the second Gaussian. Default is 4. Units are in Hartree
                energy.
            **min_cx1 (float)**: The minimum possible position of the center 
                along the x axis of the first Gaussian. Default is -8 a.u.
            **max_cx1 (float)**: The maximum possible position of the center 
                along the x axis of the first Gaussian. Default is 8 a.u.
            **min_cy1 (float)**: The minimum possible position of the center 
                along the y axis of the first Gaussian. Default is -8 a.u.
            **max_cy1 (float)**: The maximum possible position of the center 
                along the y axis of the first Gaussian. Default is 8 a.u.
            **min_cx2 (float)**: The minimum possible position of the center 
                along the x axis of the second Gaussian. Default is -8 a.u.
            **max_cx2 (float)**: The maximum possible position of the center 
                along the x axis of the second Gaussian. Default is 8 a.u.
            **min_cy2 (float)**: The minimum possible position of the center 
                along the y axis of the second Gaussian. Default is -8 a.u.
            **max_cy2 (float)**: The maximum possible position of the center 
                along the y axis of the second Gaussian. Default is 8 a.u.
            **min_kx1 (float)**: The minimum possible value for the constant
                that determines the width along the x axis of the first 
                Gaussian. Default is 1.6.
            **max_kx1 (float)**: The maximum possible value for the constant
                that determines the width along the x axis of the first 
                Gaussian. Default is 8.
            **min_ky1 (float)**: The minimum possible value for the constant
                that determines the width along the y axis of the first 
                Gaussian. Default is 1.6.
            **max_ky1 (float)**: The maximum possible value for the constant
                that determines the width along the y axis of the first 
                Gaussian. Default is 8.
            **min_kx2 (float)**: The minimum possible value for the constant
                that determines the width along the x axis of the second 
                Gaussian. Default is 1.6.
            **max_kx2 (float)**: The maximum possible value for the constant
                that determines the width along the x axis of the second 
                Gaussian. Default is 8.
            **min_ky2 (float)**: The minimum possible value for the constant
                that determines the width along the y axis of the second 
                Gaussian. Default is 1.6.
            **max_ky2 (float)**: The maximum possible value for the constant
                that determines the width along the y axis of the second 
                Gaussian. Default is 8.

        Returns:
            A dictionary of randomly generated double inverted Gaussian
            parameters.
        """

        assert min_a1 > 0
        assert min_a2 > 0
        assert min_a1 <= max_a1
        assert min_a2 <= max_a2
        assert min_cx1 <= max_cx1
        assert min_cy1 <= max_cy1
        assert min_cx2 <= max_cx2
        assert min_cy2 <= max_cy2
        assert min_kx1 <= max_kx1
        assert min_ky1 <= max_ky1
        assert min_kx2 <= max_kx2
        assert min_ky2 <= max_ky2

        # Rescale arguments
        min_cx1 = self.rescale_by_size(min_cx1)
        min_cy1 = self.rescale_by_size(min_cy1)
        min_cx2 = self.rescale_by_size(min_cx2)
        min_cy2 = self.rescale_by_size(min_cy2)
        min_kx1 = self.rescale_by_size(min_kx1)
        min_ky1 = self.rescale_by_size(min_ky1)
        min_kx2 = self.rescale_by_size(min_kx2)
        min_ky2 = self.rescale_by_size(min_ky2)

        # Sample parameters
        a1 = np.random.uniform(low=min_a1, high=max_a1)
        a2 = np.random.uniform(low=min_a2, high=max_a2)
        c_x1 = np.random.uniform(low=min_cx1, high=max_cx1)
        c_y1 = np.random.uniform(low=min_cy1, high=max_cy1)
        c_x2 = np.random.uniform(low=min_cx2, high=max_cx2)
        c_y2 = np.random.uniform(low=min_cy2, high=max_cy2)
        k_x1 = np.random.uniform(low=min_kx1, high=max_kx1)
        k_y1 = np.random.uniform(low=min_ky1, high=max_ky1)
        k_x2 = np.random.uniform(low=min_kx2, high=max_kx2)
        k_y2 = np.random.uniform(low=min_ky2, high=max_ky2)
        
        return {'a1': a1, 'a2': a2, 'c_x1': c_x1, 'c_y1': c_y1,
                'c_x2': c_x2, 'c_y2': c_y2, 'k_x1': k_x1, 'k_y1': k_y1,
                'k_x2': k_x2, 'k_y2': k_y2}
    
    def rand(self, min_k=2, max_k=7, min_r=80, max_r=180, 
             min_sig1=6, max_sig1=10, min_sig2=10, max_sig2=16, 
             p_range=[1, 2]):
        """
        Random parameter method that generates parameters for random 
        potentials.
        
        Args:
            **min_k (int)**: Determines the low end of the number of points
                used to create the convex hull. Default is 2.
            **max_k (int)**: Determines the high end of the number of points
                used to create the convex hull. Default is 7.
            **min_r (float)**: The minimum possible resolution size of the
                random blob. Default is 80 pixels (scaled for 256 x 256).
            **max_r (float)**: The maximum possible resolution size of the
                random blob. Default is 180 pixels (scaled for 256 x 256).
            **min_sig1 (float)**: The minimum possible variance of the first
                Gaussian blur. Default is 6.
            **max_sig1 (float)**: The maximum possible variance of the first
                Gaussian blur. Default is 10.
            **min_sig2 (float)**: The minimum possible variance of the second
                Gaussian blur. Default is 10.
            **max_sig2 (float)**: The maximum possible variance of the second
                Gaussian blur. Default is 16.
            **p_range (list)**: The range of exponents to sample from when
                exponentiating the potential for contrasting. Default is
                [1, 2].

        Returns:
            A dictionary of randomly generated random potential parameters.
        """

        assert min_k <= max_k
        assert min_r <= max_r
        assert min_sig1 <= max_sig1
        assert min_sig1 > 0
        assert min_sig2 <= max_sig2
        assert min_sig2 > 0
        assert all([isinstance(i, int) for i in p_range])

        # Rescale arguments
        min_r = self.rescale_by_grid(min_r)
        max_r = self.rescale_by_grid(max_r)
        min_sig1 = self.rescale_by_grid(min_sig1)
        max_sig1 = self.rescale_by_grid(max_sig1)
        min_sig2 = self.rescale_by_grid(min_sig2)
        max_sig2 = self.rescale_by_grid(max_sig2)

        # Sample parameters
        k = np.random.randint(low=min_k, high=max_k+1)
        r = np.random.uniform(low=min_r, high=max_r)
        sig1 = np.random.uniform(low=min_sig1, high=max_sig1)
        sig2 = np.random.uniform(low=min_sig2, high=max_sig2)
        p = int(np.random.choice(p_range))
        
        return {'k': k, 'r': r, 'sig1': sig1, 'sig2': sig2, 'p': p}
    
    def coulomb(self, min_z=1, max_z=118):
        """
        Random parameter method that generates parameters for Coulomb 
        potentials.
        
        Args:
            **min_z (int)**: The minimum number of allowable protons.
            **max_z (int)**: The maximum number of allowable protons.

        Returns:
            A dictionary of randomly generated Coulomb potential parameters.
        """

        assert min_z <= max_z
        assert min_z >= 1

        # Sample parameters
        z = np.random.randint(low=min_z, high=max_z)
        
        return {'z': z}
