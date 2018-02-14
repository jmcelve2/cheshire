# Cheshire

```
Alice: Would you tell me, please, which way I ought to go from here?
The Cheshire Cat: That depends a good deal on where you want to get to.
```

## What Cheshire does

This package contains functionality for generating homogeneous 1D and 2D quantum potentials and numerically solving the Schrodinger equation. The 2D potentials are generated using the methodology outlined in [this paper.](https://arxiv.org/pdf/1702.01361.pdf)

Supported 1D potentials:
* Infinite square well with interacting electrons

Supported 2D potentials:
* Infinite square well
* Simple harmonic oscillator
* Double inverted Gaussian
* Random
* Coulomb (experimental)

Example 2D potentials from the four standard potential methods:

![2D potential](https://raw.github.com/jmcelve2/cheshire/master/examples/potential_2d.png)

## How to install Cheshire

To install, pull the repo and execute the following:
```
cd cheshire
pip install .
```

## How to run Cheshire

The following code should execute if the installation works. This code will  generate a random infinite well potential, plot it, and then solve the Schrodinger equation. The resulting energy of the ground state will be printed, and the probability density of the ground state solution will be plotted.

First, import the relevant functionality:
```
from cheshire.Schrodinger import Schrodinger
from cheshire.ParamSampler import ParamSampler2D
from cheshire.Potential import PotentialFactory2D

import seaborn as sns
```

Next, initialize objects of the `PotentialFactory` and `ParamSampler` family of classes. 

In the 1D case:
```
grid = {'n_x': 128}
factory = PotentialFactory1D(**grid)
sampler = ParamSampler1D(**grid)
```

In the 2D case:
```
grid = {'n_x': 128, 'n_y': 128}
factory = PotentialFactory2D(**grid)
sampler = ParamSampler2D(**grid)
```
The `PotentialFactory` class generates objects of the class `Potential`. 

The `ParamSampler` class generates random parameters in order to generate an appropriate `Potential`. 

`Potential` objects have two attributes: `potential`, the grid of potential energy values, and `dist`, the distance between grid points. Create the potential object using parameters generated from the sampler class. Each call of `sampler` generates new parameters.
```
potential = factory.iw(**sampler.iw())
```

In the 1D case:
```
sns.tsplot(potential.potential)
```

In the 2D case:
```
sns.heatmap(potential.potential)
```

The `Schrodinger` class takes the `potential` passed to the constructor and infers the dimensionality of the problem. The appropriate solver function is then internally implemented.
```
schrodinger = Schrodinger(potential=potential, k=1)
```

Now solve the Schrodinger equation for this potential. Setting `k=1` returns a single eigenvalue and eigenvector. Internally, `scipy.sparse.linalg.eigs` is used and `which='SM'` is hardcoded to ensure that the eigenvalues and eigenvectors are returned and sorted from lowest to highest energy, e.g. if `k=1`, the ground state energy and state are returned, whereas if `k=2`, the both the ground state and first excited energies and states are returned.
```
energy, psi = schrodinger.solve()
```

Finally, plot the probability density of the ground state solution to the Schrodinger equation.

In the 1D case, marginalization should first be done:
```
prob = abs(psi[0])**2
prob = prob.sum(axis=1)
sns.tsplot(prob)
```

In the 2D case:
```
prob = abs(psi[0])**2
sns.heatmap(prob)
```