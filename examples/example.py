# The following code should execute if the installation works. This code will 
# generate a random infinite well potential, plot it, and then solve the 
# Schrodinger equation. The resulting energy of the ground state will be printed, 
# and the probability density of the ground state solution will be plotted.

# First, import the relevant functionality:
from cheshire.Schrodinger import schrodinger
from cheshire.ParamSampler import ParamSampler
from cheshire.Potential import PotentialFactory

import seaborn as sns

# Next, initialize objects of the `PotentialFactory` and `ParamSampler` classes.
grid = {'n_x': 128, 'n_y': 128}
factory = PotentialFactory(**grid)
sampler = ParamSampler(**grid)

# The `PotentialFactory` class generates objects of the class `Potential`. 
# `Potential` objects have two attributes: `potential`, the grid of potential 
# energy values, and `dist`, the distance between grid points.

# Create the potential object using parameters generated from the sampler class. 
# Each call of `sampler` generates new parameters.
potential = factory.iw(**sampler.iw())
sns.heatmap(potential.potential)

# Now solve the Schrodinger equation for this potential. Setting `k=1` returns a 
# single eigenvalue and eigenvector. Internally, `scipy.sparse.linalg.eigs` is 
# used and `which='SM'` is hardcoded to ensure that the eigenvalues and 
# eigenvectors are returned and sorted from lowest to highest energy, e.g. if 
# `k=1`, the ground state energy and state are returned, whereas if `k=2`, the 
# both the ground state and first excited energies and states are returned.
E, psi = schrodinger(potential=potential, k=1)

# Finally, plot the probability density of the ground state solution to the 
# Schrodinger equation.
prob = abs(psi[0])**2
sns.heatmap(abs(psi[0])**2)
