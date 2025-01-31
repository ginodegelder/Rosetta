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
    plt.pcolormesh(X - off, Y, hist,  cmap='viridis', vmin=0., vmax=0.2)
    plt.plot(mean - off, y_n, '--r', label="Mean")
    plt.plot(x_obs, y_obs, 'k', label='Observed Topo')
    plt.plot(x_n[best, :] - off, y_n, '--y', label="Best-Fit")
    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlabel('Distance along profile (m)', fontsize=fs)
    plt.ylabel('Elevation (m)', fontsize=fs)
    #plt.xlim([x_obs[0]-100, x_obs[0]+len(x_obs)])
    #plt.ylim([-25, 100])
    DefaultSize = fig.get_size_inches()
    fig.set_size_inches((DefaultSize[0]*fig_rescale[0], DefaultSize[1]*fig_rescale[1]),
                         forward=True)
    plt.tight_layout()

    cbaxes = inset_axes(ax, width="40%", height="5%", loc=2, borderpad=1)
    cbar = plt.colorbar(cax=cbaxes, orientation='horizontal')
    cbar.set_label("Probability", fontsize=fs-4)

    ax.legend(loc=4)
    # plt.savefig("Figs/Profiles.png", dpi=300)
    plt.savefig("Figs/Fig2b/Profiles.pdf")

    # fig2=plt.figure()
    # plt.plot(x, mean, 'r')
    # plt.plot(x, mean+std, '--k')
    # plt.plot(x, mean-std, '--k')
    # plt.xlabel('Distance along profile (m)', fontsize=fs)
    # plt.ylabel('Elevation (m)', fontsize=fs)
    # plt.tight_layout()
    # ax = plt.gca()
    # plt.savefig("Figs/Profile_mean_std.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')

    fig2=plt.figure()
    plt.plot(p025 - off, y_n, '--g')
    plt.plot(p25 - off, y_n, '--b')
    plt.plot(p50 - off, y_n, 'r')
    plt.plot(p75 - off, y_n, '--b')
    plt.plot(p975 - off, y_n, '--g')
    plt.plot(x_obs, y_obs, 'k', label='Observed Topo')
    plt.plot(x_n[best, :] - off, y_n, '--y', label="Best-Fit")
    plt.xlabel('Distance along profile (m)', fontsize=fs)
    plt.ylabel('Elevation (m)', fontsize=fs)
    #plt.xlim([x_obs[0]-100, x_obs[0] + len(x_obs)])
    #plt.ylim([-25, 100])
    plt.tight_layout()
    ax = plt.gca()
    plt.legend([2.5, 25, 50, 75, 97.5, 'Obs', 'Best-Fit'])
    # plt.savefig("Figs/Profile_median_percentiles.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig("Figs/Fig2b/Profile_median_percentiles.pdf")

    return fig, fig2
#plt.savefig("Figs/traces_hist.eps", dpi=300)

# fig = plt.figure()
# ax = plt.gca()
# plt.pcolormesh(X, Y, range_only,  cmap='viridis', vmin=0., vmax=None)
# plt.plot(x, mean, 'r')
# ax.tick_params(axis='both', which='major', labelsize=fs)
# plt.xlabel('Time (s)', fontsize=fs)
# plt.ylabel('Displacement', fontsize=fs)
# #cbar = plt.colorbar()
# #cbar.set_label("Probability", fontsize=fs)
# #ax.set_xlim(xlim)
# #ax.set_ylim(ylim)
# DefaultSize = fig.get_size_inches()
# fig.set_size_inches((DefaultSize[0]*fig_rescale[0], DefaultSize[1]*fig_rescale[1]),
#                      forward=True)
# plt.tight_layout()
# plt.savefig("traces_range.png", dpi=300)
#
# fig=plt.figure()
# plt.plot(x, y[0])
# plt.xlabel('Time (s)', fontsize=fs)
# plt.ylabel('Displacement (m)', fontsize=fs)
# plt.tight_layout()
# ax = plt.gca()
# plt.savefig("trace_1.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
# plt.savefig("trace_1.eps", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
#

# plt.show()

def sealevel(xsl, ysl, best):
    # First order statistics
    mean = np.mean(ysl, axis=0)
    std = np.std(ysl, axis=0)
    p025 = np.percentile(ysl[:, :], 2.5, axis=0)
    p25 = np.percentile(ysl[:, :], 25, axis=0)
    p50 = np.percentile(ysl[:, :], 50, axis=0)
    p75 = np.percentile(ysl[:, :], 75, axis=0)
    p975 = np.percentile(ysl[:, :], 97.5, axis=0)

    # Read the original sea level curve
    treal, ereal = tools.readfile('SL/NodeCurve153.dat')

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
    ybins = np.linspace(ymin, ymax, 175)
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
    plt.pcolormesh((-X) + 152, Y, hist, cmap='viridis', vmin=0., vmax=0.1)
    plt.plot((-xsl) + 152, mean, '--r', label="Mean")
    plt.plot(treal, ereal, 'k', label="Input SL curve")
    plt.plot((-xsl) + 152, ysl[best, :], '--y', label="Best-Fit")
    plt.gca().add_patch(Rectangle((21, 0), 9, -100, linewidth=1, edgecolor='r', facecolor='none'))
    plt.gca().add_patch(Rectangle((45, 0), 9, -100, linewidth=1, edgecolor='r', facecolor='none'))
    plt.gca().add_patch(Rectangle((55, 0), 9, -100, linewidth=1, edgecolor='r', facecolor='none'))
    plt.gca().add_patch(Rectangle((75, 0), 9, -100, linewidth=1, edgecolor='r', facecolor='none'))
    plt.gca().add_patch(Rectangle((85, 0), 9, -100, linewidth=1, edgecolor='r', facecolor='none'))
    plt.gca().add_patch(Rectangle((95, 0), 9, -100, linewidth=1, edgecolor='r', facecolor='none'))
    plt.gca().add_patch(Rectangle((105, 0), 9, -100, linewidth=1, edgecolor='r', facecolor='none'))
    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlabel('Age (ka)', fontsize=fs)
    plt.ylabel('Sea-Level (m)', fontsize=fs)
    # cbar = plt.colorbar()
    # cbar.set_label("Probability", fontsize=fs)
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    DefaultSize = fig.get_size_inches()
    fig.set_size_inches((DefaultSize[0] * fig_rescale[0], DefaultSize[1] * fig_rescale[1]),
                        forward=True)
    plt.tight_layout()
    cbaxes = inset_axes(ax, width="40%", height="5%", loc=1, borderpad=1)
    cbar = plt.colorbar(cax=cbaxes, orientation='horizontal')
    cbar.set_label("Probability", fontsize=fs - 4)
    ax.legend(loc=4)
    # plt.savefig("Figs/Sea-Level.png", dpi=300)
    plt.savefig("Figs/Fig2b/Sea-Level.pdf")

    fig2 = plt.figure()
    plt.plot((-xsl) + 152, p025, '--g')
    plt.plot((-xsl) + 152, p25, '--b')
    plt.plot((-xsl) + 152, p50, 'r')
    plt.plot((-xsl) + 152, p75, '--b')
    plt.plot((-xsl) + 152, p975, '--g')
    plt.plot(treal, ereal, 'k')
    plt.plot((-xsl) + 152, ysl[best, :], '--y')
    plt.xlabel('Age (ka)', fontsize=fs)
    plt.ylabel('Sea-level (m)', fontsize=fs)
    plt.tight_layout()
    ax = plt.gca()
    plt.legend([2.5, 25, 50, 75, 97.5, 'Input SL Curve', 'Best-Fit'])
    # plt.savefig("Figs/Sea-Level_median_percentiles.png", dpi=300)
    plt.savefig("Figs/Fig2b/Sea-Level_median_percentiles.pdf")

    return fig, fig2

    # fig = plt.figure()
    # ax = plt.gca()
    # plt.pcolormesh(X, Y, range_only,  cmap='viridis', vmin=0., vmax=None)
    # plt.plot(xsl, mean, 'r')
    # ax.tick_params(axis='both', which='major', labelsize=fs)
    # plt.xlabel('Time (s)', fontsize=fs)
    # plt.ylabel('Displacement', fontsize=fs)
    # #cbar = plt.colorbar()
    # #cbar.set_label("Probability", fontsize=fs)
    # #ax.set_xlim(xlim)
    # #ax.set_ylim(ylim)
    # DefaultSize = fig.get_size_inches()
    # fig.set_size_inches((DefaultSize[0]*fig_rescale[0], DefaultSize[1]*fig_rescale[1]),
    #                      forward=True)
    # plt.tight_layout()
    # plt.savefig("traces_range.png", dpi=300)
    #
    # fig=plt.figure()
    # plt.plot(xsl, ysl[0])
    # plt.xlabel('Time (s)', fontsize=fs)
    # plt.ylabel('Displacement (m)', fontsize=fs)
    # plt.tight_layout()
    # ax = plt.gca()
    # plt.savefig("trace_1.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    # plt.savefig("trace_1.eps", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    #
    # fig=plt.figure()
    # plt.plot(xsl, mean, 'r')
    # plt.plot(xsl, mean+std, '--k')
    # plt.plot(xsl, mean-std, '--k')
    # plt.xlabel('Time', fontsize=fs)
    # plt.ylabel('Displacement', fontsize=fs)
    # plt.tight_layout()
    # ax = plt.gca()
    # plt.savefig("traces_mean_std.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
