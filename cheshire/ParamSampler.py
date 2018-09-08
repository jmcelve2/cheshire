
import math
import numpy as np
from cheshire.Grid import *


class InfiniteWell1DSampler(Grid1D):
    """
    Class that generates parameters for the 1D infinite well.
    """

    def sample_params(self, min_l=5, max_l=23, min_c=-8, max_c=8):
        """
        Random parameter method that generates parameters for infinite square wells.

            **Args:**
                **min_l (float)**: The minimum possible allowed length for the well size. Default is 5.

                **max_l (float)**: The maximum possible allowed length for the well size. Default is 23.

                **min_c (float)**: The minimum possible center point for the well. Default is -8.

                **max_c (float)**: The maximum possible center point for the well. Default is 8.

            **Returns:**
                A dictionary of randomly generated infinite well parameters.
        """

        assert min_l <= max_l
        assert min_c <= max_c

        # Rescale arguments based on the size of the grid
        min_l = self.rescale_by_size(min_l)
        max_l = self.rescale_by_size(max_l)
        min_c = self.rescale_by_size(min_c)
        max_c = self.rescale_by_size(max_c)

        l_x = np.random.uniform(low=min_l*self.x_max/20, high=max_l*self.x_max/20)
        c_x = np.random.uniform(low=min_c, high=max_c)

        return dict(l_x=l_x, c_x=c_x)


class InfiniteWell2DSampler(Grid2D):
    """
    Class that generates parameters for the 2D infinite well.
    """

    def sample_params(self, min_l=4, max_l=15, min_c=-8, max_c=8):
        """
        Random parameter method that generates parameters for 2D infinite square wells.

            **Args:**
                **min_l (float)**: The minimum possible allowed length for the well size. Default is 4.

                **max_l (float)**: The maximum possible allowed length for the well size. Default is 15.

                **min_c (float)**: The minimum possible center point for the well. Default is -8.

                **max_c (float)**: The maximum possible center point for the well. Default is 8.

            **Returns:**
                A dictionary of randomly generated infinite well parameters.
        """

        assert min_l <= max_l
        assert min_c <= max_c

        # Rescale arguments based on the size of the grid
        min_l = self.rescale_by_size(min_l)
        max_l = self.rescale_by_size(max_l)
        min_c = self.rescale_by_size(min_c)
        max_c = self.rescale_by_size(max_c)

        while True:
            l_x, l_y = self.__sample__(min_l=min_l, max_l=max_l)
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

        return dict(l_x=l_x, l_y=l_y, c_x=c_x, c_y=c_y)

    def __sample__(self, min_l, max_l):
        """
        Sample parameters. Sampling from the energy first ensures that the desired
        energy range is obtained
        """

        energy = np.random.uniform(low=0, high=0.4)
        l_x = np.random.uniform(low=min_l*self.x_max/20, high=max_l*self.x_max/20)

        if 2*energy/math.pi**2 <= 1/(l_x**2):
            l_y = np.nan
        elif 2*energy/math.pi**2 <= 1/(l_x**2) > max_l:
            l_y = np.nan
        elif 2*energy/math.pi**2 <= 1/(l_x**2) < min_l:
            l_y = np.nan
        else:
            l_y = 1 / np.sqrt((2*energy)/(math.pi**2) - 1/(l_x**2))
        return l_x, l_y


class SimpleHarmonicOscillator2DSampler(Grid2D):
    """
    Class that generates parameters for the 2D simple harmonic oscillator.
    """

    def sample_params(self, min_kx=0, max_kx=0.16, min_ky=0, max_ky=0.16, min_cx=-8, max_cx=8, min_cy=-8, max_cy=8):
        """
        Random parameter method that generates parameters for simple harmonic oscillator potentials.

            **Args:**
                **min_kx (float)**: The minimum possible value for the well width along the x axis. Default is 0.

                **max_kx (float)**: The maximum possible value for the well width along the x axis. Default is 0.16.

                **min_ky (float)**: The minimum possible value for the well width along the y axis. Default is 0.

                **max_ky (float)**: The maximum possible value for the well width along the y axis. Default is 0.16.

                **min_cx (float)**: The minimum possible value for the center (in a.u.) of the simple harmonic oscillator
                along the x axis. Default is -8.

                **max_cx (float)**: The maximum possible value for the center (in a.u.) of the simple harmonic oscillator
                along the x axis. Default is 8.

                **min_cy (float)**: The minimum possible value for the center (in a.u.) of the simple harmonic oscillator
                along the y axis. Default is -8.

                **max_cy (float)**: The maximum possible value for the center (in a.u.) of the simple harmonic oscillator
                along the y axis. Default is 8.

            **Returns:**
                A dictionary of randomly generated simple harmonic oscillator parameters.
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

        return dict(k_x=k_x, k_y=k_y, c_x=c_x, c_y=c_y)


class DoubleInvertedGaussian2DSampler(Grid2D):
    """
    Class that generates parameters for the 2D double inverted Gaussian.
    """

    def sample_params(self, min_a1=2, max_a1=4, min_a2=2, max_a2=4, min_cx1=-8, max_cx1=8, min_cy1=-8, max_cy1=8,
                      min_cx2=-8, max_cx2=8, min_cy2=-8, max_cy2=8, min_kx1=1.6, max_kx1=8, min_ky1=1.6, max_ky1=8,
                      min_kx2=1.6, max_kx2=8, min_ky2=1.6, max_ky2=8):
        """
        Random parameter method that generates parameters for double inverted Gaussian potentials.

            **Args:**
                **min_a1 (float)**: The minimum possible value of the amplitude of the first Gaussian. Default is 2. Units
                are in Hartree energy.

                **max_a1 (float)**: The maximum possible value of the amplitude of the first Gaussian. Default is 4. Units
                are in Hartree energy.

                **min_a2 (float)**: The minimum possible value of the amplitude of the second Gaussian. Default is 2. Units
                are in Hartree energy.

                **max_a2 (float)**: The maximum possible value of the amplitude of the second Gaussian. Default is 4. Units
                are in Hartree energy.

                **min_cx1 (float)**: The minimum possible position of the center along the x axis of the first Gaussian.
                Default is -8 a.u.

                **max_cx1 (float)**: The maximum possible position of the center along the x axis of the first Gaussian.
                Default is 8 a.u.

                **min_cy1 (float)**: The minimum possible position of the center along the y axis of the first Gaussian.
                Default is -8 a.u.

                **max_cy1 (float)**: The maximum possible position of the center along the y axis of the first Gaussian.
                Default is 8 a.u.

                **min_cx2 (float)**: The minimum possible position of the center along the x axis of the second Gaussian.
                Default is -8 a.u.

                **max_cx2 (float)**: The maximum possible position of the center along the x axis of the second Gaussian.
                Default is 8 a.u.

                **min_cy2 (float)**: The minimum possible position of the center along the y axis of the second Gaussian.
                Default is -8 a.u.

                **max_cy2 (float)**: The maximum possible position of the center along the y axis of the second Gaussian.
                Default is 8 a.u.

                **min_kx1 (float)**: The minimum possible value for the constant that determines the width along the x axis
                of the first Gaussian. Default is 1.6.

                **max_kx1 (float)**: The maximum possible value for the constant that determines the width along the x axis
                of the first Gaussian. Default is 8.

                **min_ky1 (float)**: The minimum possible value for the constant that determines the width along the y axis
                of the first Gaussian. Default is 1.6.

                **max_ky1 (float)**: The maximum possible value for the constant that determines the width along the y axis
                of the first Gaussian. Default is 8.

                **min_kx2 (float)**: The minimum possible value for the constant that determines the width along the x axis
                of the second Gaussian. Default is 1.6.

                **max_kx2 (float)**: The maximum possible value for the constant that determines the width along the x axis
                of the second Gaussian. Default is 8.

                **min_ky2 (float)**: The minimum possible value for the constant that determines the width along the y axis
                of the second Gaussian. Default is 1.6.

                **max_ky2 (float)**: The maximum possible value for the constant that determines the width along the y axis
                of the second Gaussian. Default is 8.

            **Returns:**
                A dictionary of randomly generated double inverted Gaussian parameters.
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

        return dict(a1=a1, a2=a2, c_x1=c_x1, c_y1=c_y1, c_x2=c_x2, c_y2=c_y2, k_x1=k_x1, k_y1=k_y1, k_x2=k_x2,
                    k_y2=k_y2)


class Random2DSampler(Grid2D):
    """
    Class that generates parameters for a random 2D potential.
    """

    def sample_params(self, min_k=2, max_k=7, min_r=80, max_r=180, min_sig1=6, max_sig1=10, min_sig2=10, max_sig2=16,
                      p_range=[0.5, 1.0, 1.5, 2.0]):
        """
        Random parameter method that generates parameters for random potentials.

            **Args:**
                **min_k (int)**: Determines the low end of the number of points used to create the convex hull. Default is
                2.

                **max_k (int)**: Determines the high end of the number of points used to create the convex hull. Default is
                7.

                **min_r (float)**: The minimum possible resolution size of the random blob. Default is 80 pixels (scaled
                for 256 x 256).

                **max_r (float)**: The maximum possible resolution size of the random blob. Default is 180 pixels (scaled
                for 256 x 256).

                **min_sig1 (float)**: The minimum possible variance of the first Gaussian blur. Default is 6.

                **max_sig1 (float)**: The maximum possible variance of the first Gaussian blur. Default is 10.

                **min_sig2 (float)**: The minimum possible variance of the second Gaussian blur. Default is 10.

                **max_sig2 (float)**: The maximum possible variance of the second Gaussian blur. Default is 16.

                **p_range (list)**: The range of exponents to sample from when exponentiating the potential for contrasting.
                Default is [1, 2].

            **Returns:**
                A dictionary of randomly generated random potential parameters.
        """

        assert min_k <= max_k
        assert min_r <= max_r
        assert min_sig1 <= max_sig1
        assert min_sig1 > 0
        assert min_sig2 <= max_sig2
        assert min_sig2 > 0
        assert all([isinstance(i, float) for i in p_range])

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
        p = np.random.choice(p_range)

        return dict(k=k, r=r, sig1=sig1, sig2=sig2, p=p)


class Coulomb2DSampler(Grid2D):
    """
    Class that generates parameters for the 2D Coulomb potential.
    """

    def sample_params(self, min_z=1, max_z=118, min_cx=-8, max_cx=8, min_cy=-8, max_cy=8, alpha=10**-9):
        """
        Random parameter method that generates parameters for Coulomb potentials.

            **Args:**
                **min_z (int)**: The minimum number of allowable protons.

                **max_z (int)**: The maximum number of allowable protons.

                **min_cx (float)**: The minimum possible value for the center (in a.u.) of the Coulomb potential along the x
                axis. Default is -8.

                **max_cx (float)**: The maximum possible value for the center (in a.u.) of the Coulomb potential along the x
                axis. Default is 8.

                **min_cy (float)**: The minimum possible value for the center (in a.u.) of the Coulomb potential along the y
                axis. Default is -8.

                **max_cy (float)**: The maximum possible value for the center (in a.u.) of the Coulomb potential along the y
                axis. Default is 8.

            **Returns:**
                A dictionary of randomly generated Coulomb potential parameters.
        """

        assert min_z <= max_z
        assert min_z >= 1

        # Sample parameters
        z = np.random.randint(low=min_z, high=max_z)
        c_x = np.random.uniform(low=min_cx, high=max_cx)
        c_y = np.random.uniform(low=min_cy, high=max_cy)
        alpha = alpha

        return dict(z=z, c_x=c_x, c_y=c_y, alpha=alpha)
