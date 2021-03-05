#!/usr/bin/env python3
import numpy as np
import routingpy
from matplotlib import pyplot as plt


def show_path(path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(path[:, 0], path[:, 1], 'x')
    ax.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    gh = routingpy.Graphhopper(base_url="http://localhost:8989")
    print(gh)
    loc_max_eith = [9.214048, 48.834593]
    loc_wernau = [9.411373, 48.695684]
    dir = gh.directions([loc_max_eith, loc_wernau], profile='bike')
    path = np.array(dir.geometry)
    show_path(path)
