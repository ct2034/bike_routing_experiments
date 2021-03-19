#!/usr/bin/env python3
import numpy as np
import routingpy
from matplotlib import pyplot as plt


def show_path(path):
    assert path.shape[1] == 3
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(path[:, 1], path[:, 2], path[:, 0])
    plt.show()


if __name__ == "__main__":
    gh = routingpy.Graphhopper(base_url="http://localhost:8989")
    print(gh)
    loc_max_eith = [9.214048, 48.834593]
    loc_wernau = [9.411373, 48.695684]
    dir = gh.directions([loc_max_eith, loc_wernau],
                        profile='bike', elevation=True)
    path = np.array(dir.geometry)
    show_path(path)
