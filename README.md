# Cheshire

```
Alice: Would you tell me, please, which way I ought to go from here?
The Cheshire Cat: That depends a good deal on where you want to get to.
```

### What Cheshire does

This package contains functionality for generating homogeneous 2D quantum potentials and numerically solving the Schrodinger equation. The potentials are generated using the methodology outlined in [this paper.](https://arxiv.org/pdf/1702.01361.pdf)

Currently, the following kinds of potentials are generated:
* Infinite square well
* Simple harmonic oscillator
* Double inverted Gaussian
* Random
* Coulomb (experimental)

### How to install Cheshire

To install, pull the repo and execute the following:
```
cd cheshire
pip install .
```

### How to run Cheshire

The following code should execute if the installation works. This code will generate a random infinite well potential, plot it, and then solve the Schrodinger equation. The resulting energy of the ground state will be printed, and the probability density of the ground state solution will be plotted.

First, import the relevant functionality:
```
from cheshire.Schrodinger import schrodinger
from cheshire.ParamSampler import ParamSampler
from cheshire.Potential import PotentialFactory

import numpy as np
import seaborn as sns
```

Next, initialize objects of the `PotentialFactory` and `ParamSampler` classes.
```
factory = PotentialFactory()
sampler = ParamSampler()
```
The `PotentialFactory` class generates objects of the class `Potential`. `Potential` objects have two attributes: `potential`, the grid of potential energy values, and `dist`, the distance between grid points.

Create the potential object using parameters generated from the sampler class. This object contains randomly generated parameters for the infinite well. Then plot the potential.
```
params = sampler.iw()
potential = factory.iw(**params)

sns.heatmap(potential.potential)
```


Now solve the Schrodinger equation for this potential. Setting `k=1` ensures that the ground state eigenvalue and eigenvector are returned. Internally, `scipy.sparse.linalg.eigs` is used and `which='SM'` is hardcoded to ensure that the eigenvalues and eigenvectors are returned low-to-high.
```
E, psi = schrodinger(potential=potential, k=1)
```

Finally, plot the probability density of the ground state solution to the Schrodinger equation.

```
sns.heatmap(abs(np.transpose(psi)[0].flatten().reshape((128, 128))))
```