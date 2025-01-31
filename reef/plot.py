import numpy as np
import matplotlib.pyplot as plt


def plot_profile(x, y, dz, xmin, xmax, ymin, ymax, e, tmax, u0, x_ini=None,
                 y_ini=None, fs=11):
    # some required consts
    upmax = u0 * tmax
    imax = int(tmax)
    # Figure
    xy = plt.subplot()  # Opens the figure
    xy.grid(color='grey', lw=0.5, linestyle='-')  # Defines color grid
    if x_ini is not None and y_ini is not None:
        xy.plot(x_ini, y_ini + upmax, color='0.8', lw=1)  # Plots first profile
    xy.set_xlabel('Distance (m)', fontsize=fs)
    xy.set_ylabel('Elevation (m)', fontsize=fs)
    xy.tick_params(labelsize=fs)
    xy.plot(x, y + dz, color='black', lw=0.5)
    xy.axis([xmin - 200, xmax + 200, ymin - 20, ymax + 20])
    xy.hlines(e[imax], x[0], x[np.argmax(e[imax] <= y)], color='blue', lw=1)
    return xy

