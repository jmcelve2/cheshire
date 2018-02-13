
from cheshire.ParamSampler import ParamSampler2D
from cheshire.Potential import PotentialFactory2D
from matplotlib import pyplot as plt
import seaborn as sns

factory = PotentialFactory2D()
sampler = ParamSampler2D()

iw1 = factory.iw(**sampler.iw()).potential
iw2 = factory.iw(**sampler.iw()).potential
iw3 = factory.iw(**sampler.iw()).potential

sho1 = factory.sho(**sampler.sho()).potential
sho2 = factory.sho(**sampler.sho()).potential
sho3 = factory.sho(**sampler.sho()).potential

dig1 = factory.dig(**sampler.dig()).potential
dig2 = factory.dig(**sampler.dig()).potential
dig3 = factory.dig(**sampler.dig()).potential

rand1 = factory.rand(**sampler.rand()).potential
rand2 = factory.rand(**sampler.rand()).potential
rand3 = factory.rand(**sampler.rand()).potential

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))

cols = ["Infinite Well",
        "Simple Harmonic Oscillator", 
        "Double Inverted Gaussian", 
        "Random"]

for ax, col in zip(axes[0], cols):
    ax.set_title(col)

sns.heatmap(iw1, ax=axes[0, 0], xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(iw2, ax=axes[1, 0], xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(iw3, ax=axes[2, 0], xticklabels=False, yticklabels=False, cbar=False)

sns.heatmap(sho1, ax=axes[0, 1], xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(sho2, ax=axes[1, 1], xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(sho3, ax=axes[2, 1], xticklabels=False, yticklabels=False, cbar=False)

sns.heatmap(dig1, ax=axes[0, 2], xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(dig2, ax=axes[1, 2], xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(dig3, ax=axes[2, 2], xticklabels=False, yticklabels=False, cbar=False)

sns.heatmap(rand1, ax=axes[0, 3], xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(rand2, ax=axes[1, 3], xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(rand3, ax=axes[2, 3], xticklabels=False, yticklabels=False, cbar=False)

plt.tight_layout()
plt.show()
plt.savefig(fname="example.png", dpi=300)
