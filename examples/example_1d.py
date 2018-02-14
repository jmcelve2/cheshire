# The following code should execute if the installation works. This code will 
# generate a random infinite well potential, plot it, and then solve the 
# Schrodinger equation. The resulting energy of the ground state will be printed, 
# and the probability density of the ground state solution will be plotted.

# First, import the relevant functionality:
from cheshire.Schrodinger import Schrodinger
from cheshire.ParamSampler import ParamSampler1D
from cheshire.Potential import PotentialFactory1D

import seaborn as sns

# Next, initialize objects of the `PotentialFactory1D and `ParamSampler1D` classes.
grid = {'n_x': 128}
factory = PotentialFactory1D(**grid)
sampler = ParamSampler1D(**grid)

# The `PotentialFactory1D` class generates objects of the class `Potential`.
# As the name suggests, these potentials are one-dimensional.
# `Potential` objects have two attributes: `potential`, the grid of potential 
# energy values, and `dist`, the distance between grid points.

# Create the potential object using parameters generated from the sampler class. 
# Each call of `sampler` generates new parameters.
potential = factory.iw(**sampler.iw())
sns.tsplot(potential.potential)

# The `Schrodinger` class takes the `potential` passed to the constructor and
# infers the dimensionality of the problem. The appropriate solver function is
# then internally implemented.
schrodinger = Schrodinger(potential=potential, k=1)

# Now solve the Schrodinger equation for this potential. Setting `k=1` returns a
# single eigenvalue and eigenvector. Internally, `scipy.sparse.linalg.eigs` is
# used and `which='SM'` is hardcoded to ensure that the eigenvalues and
# eigenvectors are returned and sorted from lowest to highest energy, e.g. if
# `k=1`, the ground state energy and state are returned, whereas if `k=2`, the
# both the ground state and first excited energies and states are returned.
energy, psi = schrodinger.solve()

# Finally, plot the probability density of the ground state solution to the 
# Schrodinger equation.
prob = abs(psi[0])**2

# Marginalize out one of the axes.
prob = prob.sum(axis=1)
sns.tsplot(prob)
