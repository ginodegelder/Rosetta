# Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from matplotlib.patches import Rectangle
from reef import tools as tools

def profile(x_n, y_n, x_obs, y_obs, best):
    #x_obs = x_obs - x_obs[0]
    off = x_n[0, 0]
    # x_n = x_n[stp:]
    # y_n = y_n[stp:]
    # x_obs = x_obs[stp:]
    # y_obs = y_obs[stp:]

    # First order statistics
    mean = np.mean(x_n, axis=0)
    std = np.std(x_n, axis=0)
    p025 = np.percentile(x_n[:, :], 2.5, axis=0)
    p25 = np.percentile(x_n[:, :], 25, axis=0)
    p50 = np.percentile(x_n[:, :], 50, axis=0)
    p75 = np.percentile(x_n[:, :], 75, axis=0)
    p975 = np.percentile(x_n[:, :], 97.5, axis=0)

    # Convert to histograms
    n_traces, n_samples = x_n.shape

    # 1- construct two vectors containing the samples to prepare 2d histogram
    # sample_t is simply n_traces times the time vectors
    # sample_a is the concatenation of all traces
    # No need to loop just use reshape !
    # Result: a point (t, amp) is (sample_t[i], sample_a[i]
    # ----------------------------------------------------
    sample_t = np.tile(y_n, n_traces)
    sample_a = x_n.flatten()

    # 2- define the bins
    # ------------------
    dy = y_n[1] - y_n[0]
    ybins = np.linspace(y_n[0] - dy/2., y_n[-1] + dy/2., n_samples + 1)
    xmin, xmax = np.amin(x_n), np.amax(x_n)
    xbins = np.linspace(xmin, xmax, 200)
    #ybins = np.linspace(-25, 125, 150)
    X, Y = np.meshgrid(xbins, ybins)

    # 3- just use np.histogram2d and shape it for plotting
    # ----------------------------------------------------
    cmin = 0.00001
    hist, xedges, yedges = np.histogram2d(sample_a, sample_t, bins=[xbins, ybins])

    range_only = np.copy(hist)
    range_only[hist > cmin] = 1.
    range_only[hist <= cmin] = 0.
    range_only = range_only.T

    hist = np.ma.array(hist, mask=hist<cmin).T/n_traces

    # Plot parameters
    #----------------

    fs=14
    fig_rescale = [1.25, 0.75]
    ##############################################################
    # PLOTS
    ##############################################################

    fig = plt.figure()
    ax = plt.gca()
    plt.pcolormesh(X - off, Y, hist, cmap='viridis', vmin=0., vmax=0.2)
    plt.plot(mean - off, y_n, '--r', label="Mean")
    plt.plot(x_obs, y_obs, 'k', label='Observed Topo')
    plt.plot(x_n[best, :] - off, y_n, '--y', label="Best-Fit")
    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlabel('Distance along profile (m)', fontsize=fs)
    plt.ylabel('Elevation (m)', fontsize=fs)
    # plt.xlim([x_obs[0]-100, x_obs[0]+len(x_obs)])
    # plt.ylim([-25, 100])
    DefaultSize = fig.get_size_inches()
    fig.set_size_inches((DefaultSize[0] * fig_rescale[0], DefaultSize[1] * fig_rescale[1]),
                        forward=True)
    plt.tight_layout()

    cbaxes = inset_axes(ax, width="40%", height="5%", loc=2, borderpad=1)
    cbar = plt.colorbar(cax=cbaxes, orientation='horizontal')
    cbar.set_label("Probability", fontsize=fs - 4)

    ax.legend(loc=4)
    plt.savefig("Figs/FigS4c/Profiles.pdf")

    fig2 = plt.figure()
    plt.plot(p025 - off, y_n, '--g')
    plt.plot(p25 - off, y_n, '--b')
    plt.plot(p50 - off, y_n, 'r')
    plt.plot(p75 - off, y_n, '--b')
    plt.plot(p975 - off, y_n, '--g')
    plt.plot(x_obs, y_obs, 'k', label='Observed Topo')
    plt.plot(x_n[best, :] - off, y_n, '--y', label="Best-Fit")
    plt.xlabel('Distance along profile (m)', fontsize=fs)
    plt.ylabel('Elevation (m)', fontsize=fs)
    # plt.xlim([x_obs[0]-100, x_obs[0] + len(x_obs)])
    # plt.ylim([-25, 100])
    plt.tight_layout()
    ax = plt.gca()
    plt.legend([2.5, 25, 50, 75, 97.5, 'Obs', 'Best-Fit'])
    plt.savefig("Figs/FigS4c/Profile_median_percentiles.pdf")

    best_prof = x_n[best, :]
    np.savetxt("Figs/FigS4c/MeanProfile.txt", mean)
    np.savetxt("Figs/FigS4c/MedianProfile.txt", p50)
    np.savetxt("Figs/FigS4c/BestProfile.txt", best_prof)

    return fig, fig2

def sealevel(xsl, ysl, best):
    # First order statistics
    mean = np.mean(ysl, axis=0)
    std = np.std(ysl, axis=0)
    p025 = np.percentile(ysl[:, :], 2.5, axis=0)
    p25 = np.percentile(ysl[:, :], 25, axis=0)
    p50 = np.percentile(ysl[:, :], 50, axis=0)
    p75 = np.percentile(ysl[:, :], 75, axis=0)
    p975 = np.percentile(ysl[:, :], 97.5, axis=0)

    # Convert to histograms
    n_traces, n_samples = ysl.shape

    # 1- construct two vectors containing the samples to prepare 2d histogram
    # sample_t is simply n_traces times the time vectors
    # sample_a is the concatenation of all traces
    # No need to loop just use reshape !
    # Result: a point (t, amp) is (sample_t[i], sample_a[i]
    # ----------------------------------------------------
    sample_t = np.tile(xsl, n_traces)
    sample_a = ysl.flatten()

    # 2- define the bins
    # ------------------
    dx = xsl[1] - xsl[0]
    xbins = np.linspace(xsl[0] - dx / 2., xsl[-1] + dx / 2., n_samples + 1)
    ymin, ymax = np.amin(ysl), np.amax(ysl)
    ybins = np.linspace(ymin, ymax, 154)
    X, Y = np.meshgrid(xbins, ybins)

    # 3- just use np.histogram2d and shape it for plotting
    # ----------------------------------------------------
    cmin = 0.00001
    hist, xedges, yedges = np.histogram2d(sample_t, sample_a, bins=[xbins, ybins])

    range_only = np.copy(hist)
    range_only[hist > cmin] = 1.
    range_only[hist <= cmin] = 0.
    range_only = range_only.T

    hist = np.ma.array(hist, mask=hist < cmin).T / n_traces

    # Plot parameters
    # ----------------

    fs = 14
    fig_rescale = [1.25, 0.75]
    ##############################################################
    # PLOTS
    ##############################################################

    fig = plt.figure()
    ax = plt.gca()
    plt.pcolormesh((-X) + 449, Y, hist, cmap='viridis', vmin=0., vmax=0.1)
    plt.plot((-xsl) + 449, p50, '-r', label="Median")
    plt.plot((-xsl) + 449, p025, '--r')
    plt.plot((-xsl) + 449, p975, '--r')
    plt.plot((-xsl) + 449, ysl[best, :], '--y', label="Best-Fit")
    plt.gca().add_patch(Rectangle((4, 6), 4, -6, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 1
    plt.gca().add_patch(Rectangle((19, -50), 8, -100, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 2
    plt.gca().add_patch(Rectangle((36, -15), 7, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 3a
    plt.gca().add_patch(Rectangle((44, -15), 5, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 3b
    plt.gca().add_patch(Rectangle((50, -15), 7, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 3c
    plt.gca().add_patch(Rectangle((58, -15), 16, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 4
    plt.gca().add_patch(Rectangle((75, -15), 10, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 5a
    plt.gca().add_patch(Rectangle((86, -15), 8, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 5b
    plt.gca().add_patch(Rectangle((95, -5), 10, -31, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 5c
    plt.gca().add_patch(Rectangle((106, -15), 11, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 5d
    plt.gca().add_patch(Rectangle((118, 9), 10, -9, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 5e
    plt.gca().add_patch(Rectangle((132, -15), 11, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 6a
    plt.gca().add_patch(Rectangle((144, -15), 11, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 6b
    plt.gca().add_patch(Rectangle((156, -15), 7, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 6c
    plt.gca().add_patch(Rectangle((164, -15), 13, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 6d
    plt.gca().add_patch(Rectangle((178, -15), 13, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 6e
    plt.gca().add_patch(Rectangle((192, 5), 9, -35, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 7a
    plt.gca().add_patch(Rectangle((202, 5), 5, -43, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 7b
    plt.gca().add_patch(Rectangle((208, 11), 11, -42, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 7c
    plt.gca().add_patch(Rectangle((220, -15), 11, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 7d
    plt.gca().add_patch(Rectangle((232, 11), 11, -47, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 7e
    plt.gca().add_patch(Rectangle((245, -15), 11, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 8a
    plt.gca().add_patch(Rectangle((257, -15), 8, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 8b
    plt.gca().add_patch(Rectangle((266, -15), 9, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 8c
    plt.gca().add_patch(Rectangle((276, -15), 10, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 9a
    plt.gca().add_patch(Rectangle((287, -15), 17, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 9b
    plt.gca().add_patch(Rectangle((305, 11), 12, -161, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 9c
    plt.gca().add_patch(Rectangle((318, 9), 3, -159, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 9d
    plt.gca().add_patch(Rectangle((322, 15), 10, -45, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 9e
    plt.gca().add_patch(Rectangle((335, -15), 13, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 10a
    plt.gca().add_patch(Rectangle((349, -15), 5, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 10b
    plt.gca().add_patch(Rectangle((355, -15), 7, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 10c
    plt.gca().add_patch(Rectangle((363, -15), 9, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 11a-1
    plt.gca().add_patch(Rectangle((373, -15), 7, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 11a-2
    plt.gca().add_patch(Rectangle((381, 9), 7, -159, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 11a-3
    plt.gca().add_patch(Rectangle((389, 9), 8, -159, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 11b
    plt.gca().add_patch(Rectangle((398, 13), 16, -19, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 11c
    plt.gca().add_patch(Rectangle((426, -15), 12, -135, linewidth=1, edgecolor='r', facecolor='none'))  # MIS 12a
    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlabel('Age (ka)', fontsize=fs)
    plt.ylabel('Sea-Level (m)', fontsize=fs)
    DefaultSize = fig.get_size_inches()
    fig.set_size_inches((DefaultSize[0] * fig_rescale[0], DefaultSize[1] * fig_rescale[1]),
                        forward=True)
    plt.tight_layout()
    cbaxes = inset_axes(ax, width="40%", height="5%", loc=1, borderpad=1)
    cbar = plt.colorbar(cax=cbaxes, orientation='horizontal')
    cbar.set_label("Probability", fontsize=fs - 4)
    ax.legend(loc=4)
    plt.savefig("Figs/FigS4c/Sea-Level.pdf")

    fig2 = plt.figure()
    plt.plot((-xsl) + 449, p025, '--g')
    plt.plot((-xsl) + 449, p25, '--b')
    plt.plot((-xsl) + 449, p50, 'r')
    plt.plot((-xsl) + 449, p75, '--b')
    plt.plot((-xsl) + 449, p975, '--g')
    plt.plot((-xsl) + 449, ysl[best, :], '--y')
    plt.xlabel('Age (ka)', fontsize=fs)
    plt.ylabel('Sea-level (m)', fontsize=fs)
    plt.tight_layout()
    ax = plt.gca()
    plt.legend([2.5, 25, 50, 75, 97.5, 'Best-Fit'])
    plt.savefig("Figs/FigS4c/Sea-Level_median_percentiles.pdf")

    return fig, fig2

