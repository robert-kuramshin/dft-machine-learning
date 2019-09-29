from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd


x_in = pd.read_csv("step/composite_feature_analysis.csv")

x = x_in["EA (eV)"].to_numpy()
y = x_in["# Li"].to_numpy()
Z = pd.read_csv("res/no-comp/krr_no_compound_test.csv")["pred(y)"].to_numpy().reshape((100,100))

X = x.reshape((100,100))
Y = y.reshape((100,100))

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

#need to descale inputs and outputs
#match against input data