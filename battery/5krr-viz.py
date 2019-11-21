from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler

def inverse_transforms(x,name):
    train = pd.read_csv("../data/battery/train.csv")

    #specify feature column names
    feature_cols = [
    "HOMO (eV)",
    "LUMO (eV)",
    "EA (eV)",
    "# C",
    "# B",
    "# O",
    "HOMO-LUMO gap",
    "# Li",
    "# H",
    "No. of Aromatic Rings",
    ]

    #splitting into dependant and independant variables
    X_train = train.loc[:, feature_cols]

    print (np.max(X_train[name]))
    print (np.min(X_train[name]))

    #normalizing 
    scaler_1 = StandardScaler()  
    scaler_1.fit(X_train)
    X_train2 = scaler_1.transform(X_train)

    scaler_2 = MaxAbsScaler()  
    scaler_2.fit(X_train2)
    X_train2 = scaler_2.transform(X_train2)
    print (np.max(X_train2[:,X_train.columns.get_loc(name)]))
    print (np.min(X_train2[:,X_train.columns.get_loc(name)]))

    df_empty = pd.DataFrame(np.zeros((10000,10)),columns=X_train.columns)

    df_empty[name] = x

    df_empty = scaler_2.inverse_transform(df_empty) 
    df_empty = scaler_1.inverse_transform(df_empty) 

    print (np.max(df_empty[:,X_train.columns.get_loc(name)]))
    print (np.min(df_empty[:,X_train.columns.get_loc(name)]))

    return df_empty[:,X_train.columns.get_loc(name)]


def main():
    x_in = pd.read_csv("step/composite_feature_analysis.csv")

    x = x_in["EA (eV)"].to_numpy()
    y = x_in["# Li"].to_numpy()

    x = inverse_transforms(x,"EA (eV)")
    y = inverse_transforms(y,"# Li")
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
    ax.set_xlabel("EA (eV)")
    ax.set_ylabel("# Li")
    ax.set_zlabel("RP (V) Predicted")
    ax.view_init(elev=39., azim=60)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


    #need to descale inputs and outputs
    #match against input data


    # train = pd.read_csv("../data/battery/train.csv")
    # x = train["EA (eV)"].to_numpy()
    # y = train["# Li"].to_numpy()
    # z = train["RP (V) - DFT"].to_numpy()


    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # surf = ax.scatter3D(x,y,z, cmap=cm.coolwarm)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    main()