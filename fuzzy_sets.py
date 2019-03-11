import numpy as np
from matplotlib import gridspec, pyplot as plt
import skfuzzy as fuzz
from curve_mf import kinked_curve_mf

from skfuzzy import trapmf
from skfuzzy import smf as s_shape_mf
from skfuzzy import zmf as z_shape_mf

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 7}
plt.rc('font', **font)

universe = np.arange(0, 13)

mfA = kinked_curve_mf(universe, [(2, 0), (8, 0.5), (9, 0.25), (14, 0)])
mfB = kinked_curve_mf(universe, [(0, 1), (5, 0)])
mfC = kinked_curve_mf(universe, [(0, 0), (3, 1), (5, 0)])

B_and_C, mf_B_and_C = fuzz.fuzzy_or(universe, mfA, universe, mfB)
A_or_B_and_C, mf_A_or_B_and_C = fuzz.fuzzy_and(B_and_C, mf_B_and_C, universe, mfC)

fig = plt.figure(figsize=(10, 8))
grid = gridspec.GridSpec(nrows=2, ncols=6)

axA  = fig.add_subplot(grid[0, :2],  xlim=(0, max(universe)), ylim=(0, 1), title='A')
axB  = fig.add_subplot(grid[0, 2:4], sharex=axA, sharey=axA, title='B')
axC  = fig.add_subplot(grid[0, 4:],  sharex=axA, sharey=axA, title='C')
axBC = fig.add_subplot(grid[1, :3],  sharex=axA, sharey=axA, title='B^C')
axD  = fig.add_subplot(grid[1, 3:],  sharex=axA, sharey=axA, title='AvB^C')

fig.tight_layout()
plt.locator_params(nbins=len(universe))

axA.plot(universe, mfA)
axB.plot(universe, mfB)
axC.plot(universe, mfC)
axBC.plot(universe, mf_B_and_C)
axD.plot(universe, mf_A_or_B_and_C)

plt.show()
