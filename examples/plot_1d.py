
from cheshire.ParamSampler import ParamSampler1D
from cheshire.Potential import PotentialFactory1D
from matplotlib import pyplot as plt
import seaborn as sns

factory = PotentialFactory1D()
sampler = ParamSampler1D()

iw1 = factory.iw(**sampler.iw()).potential
iw2 = factory.iw(**sampler.iw()).potential
iw3 = factory.iw(**sampler.iw()).potential

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

cols = ["Infinite Well"]

for ax, col in zip(axes[0], cols):
    ax.set_title(col)

sns.heatmap(iw1, ax=axes[0, 0], xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(iw2, ax=axes[1, 0], xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(iw3, ax=axes[2, 0], xticklabels=False, yticklabels=False, cbar=False)

plt.tight_layout()
plt.show()
plt.savefig(fname="potential_1d.png", dpi=300)
