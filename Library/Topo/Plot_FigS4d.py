# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:34:25 2024

@author: Yanni
"""

import os
import sys
# sys.path.append("/home/nhedjazi/src/sealevel")
sys.path.insert(0, os.path.abspath('../../'))

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#from scipy import stats
from matplotlib.patches import Rectangle
from Inputs import sea_level
#from reef import tools as tools

def profile_x(x_n, y_n, x_obs, y_obs, best, i, path):
    # CHANGE EVERYTHIGN TO Y ?
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
    plt.savefig(path + "/Profiles_{}.pdf".format(i))

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
    plt.savefig(path + "/Profile_median_percentiles_{}.pdf".format(i))

    best_prof = x_n[best, :]
    np.savetxt(path + "/MeanProfile_{}.txt".format(i), mean)
    np.savetxt(path + "/MedianProfile_{}.txt".format(i), p50)
    np.savetxt(path + "/BestProfile_{}.txt".format(i), best_prof)

    return fig, fig2


def profile_y(x_n, y_n, x_obs, y_obs, best, i, path):
    # CHANGE EVERYTHIGN TO Y ?
    #x_obs = x_obs - x_obs[0]
    #off = y_n[0, 0]
    # x_n = x_n[stp:]
    # y_n = y_n[stp:]
    # x_obs = x_obs[stp:]
    # y_obs = y_obs[stp:]

    # # First order statistics
    # mean = np.mean(y_n, axis=0)
    # std = np.std(y_n, axis=0)
    # p025 = np.percentile(y_n[:, :], 2.5, axis=0)
    # p25 = np.percentile(y_n[:, :], 25, axis=0)
    # p50 = np.percentile(y_n[:, :], 50, axis=0)
    # p75 = np.percentile(y_n[:, :], 75, axis=0)
    # p975 = np.percentile(y_n[:, :], 97.5, axis=0)
    mean = np.nanmean(y_n, axis=0)
    std = np.nanstd(y_n, axis=0)
    p025 = np.nanpercentile(y_n[:, :], 2.5, axis=0)
    p25 = np.nanpercentile(y_n[:, :], 25, axis=0)
    p50 = np.nanpercentile(y_n[:, :], 50, axis=0)
    p75 = np.nanpercentile(y_n[:, :], 75, axis=0)
    p975 = np.nanpercentile(y_n[:, :], 97.5, axis=0)

    # Convert to histograms
    n_traces, n_samples = y_n.shape

    # 1- construct two vectors containing the samples to prepare 2d histogram
    # sample_t is simply n_traces times the time vectors
    # sample_a is the concatenation of all traces
    # No need to loop just use reshape !
    # Result: a point (t, amp) is (sample_t[i], sample_a[i]
    # ----------------------------------------------------
    # sample_t = np.tile(x_n, n_traces)
    # sample_a = y_n.flatten()
    # 1. Filter out NaN values from y_n and their corresponding x_n entries
    valid_mask = ~np.isnan(y_n).any(axis=1)    
    valid_y_n = y_n[valid_mask]
    valid_x_n = np.tile(x_n, valid_y_n.shape[0])

    # 2. Flattening the valid arrays for 2D histogram
    sample_t = valid_x_n
    sample_a = valid_y_n.flatten()

    # 2- define the bins
    # ------------------
    dx = x_n[1] - x_n[0]
    xbins = np.linspace(x_n[0] - dx/2., x_n[-1] + dx/2., n_samples + 1)
    #y_concat = np.hstack((y_obs, y_n))
    ymin_obs, ymax_obs = np.amin(y_obs), np.amax(y_obs)
    #ymin_n, ymax_n = np.amin(y_n), np.amax(y_n) modif heterogen
    ymin_n, ymax_n = np.nanmin(y_n), np.nanmax(y_n)
    ymin, ymax = min(ymin_obs, ymin_n), max(ymax_obs, ymax_n)
    ybins = np.linspace(ymin, ymax + (ymax-ymin)*0.3, 200)
    #ybins = np.linspace(-25, 125, 150)
    X, Y = np.meshgrid(xbins, ybins)

    # 3- just use np.histogram2d and shape it for plotting
    # ----------------------------------------------------
    cmin = 0.00001
    # hist, xedges, yedges = np.histogram2d(sample_a, sample_t, bins=[xbins, ybins])
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
    plt.pcolormesh(X, Y, hist, cmap='viridis', vmin=0., vmax=0.2)
    plt.plot(x_n, mean, '--r', label="Mean")
    plt.plot(x_obs, y_obs, 'k', label='Observed Topo')
    plt.plot(x_n, y_n[best, :], '--y', label="Best-Fit")
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
    plt.savefig(path + "/Profiles_{}.pdf".format(i))

    fig2 = plt.figure()
    plt.plot(x_n, p025, '--g')
    plt.plot(x_n, p25, '--b')
    plt.plot(x_n, p50, 'r')
    plt.plot(x_n, p75, '--b')
    plt.plot(x_n, p975, '--g')
    plt.plot(x_obs, y_obs, 'k', label='Observed Topo')
    plt.plot(x_n, y_n[best, :], '--y', label="Best-Fit")
    plt.xlabel('Distance along profile (m)', fontsize=fs)
    plt.ylabel('Elevation (m)', fontsize=fs)
    # plt.xlim([x_obs[0]-100, x_obs[0] + len(x_obs)])
    # plt.ylim([-25, 100])
    plt.tight_layout()
    ax = plt.gca()
    plt.legend([2.5, 25, 50, 75, 97.5, 'Obs', 'Best-Fit'])
    plt.savefig(path + "/Profile_median_percentiles_{}.pdf".format(i))

    best_prof = y_n[best, :]
    np.savetxt(path + "/MeanProfile_{}.txt".format(i), mean)
    np.savetxt(path + "/MedianProfile_{}.txt".format(i), p50)
    np.savetxt(path + "/BestProfile_{}.txt".format(i), best_prof)

    return fig, fig2


# Creates the SL dataframe
df_SL = pd.DataFrame.from_dict(sea_level, orient = "index", 
                               columns = ['t_min', 't_max', 'step_t', 
                                          'e_start', 'e_min', 'e_max', 
                                          'step_e'])

# Puts t_start in first column, to have normal indexes.
df_SL.insert(0, 't_start', df_SL.index)
df_SL.reset_index(drop = True, inplace = True)
df_SL_free = pd.DataFrame(columns = df_SL.columns)  

for i in df_SL.index:
    # Extracts the fixed SL nodes and puts them in "df_SL_fixed" dataframe
    if df_SL.loc[i].isnull().values.any() == False:
        df_SL_free.loc[i] = df_SL.loc[i]
    # Extracts the free SL nodes and puts them in "df_SL_free" dataframe
    else:
        continue
    

def sealevel(xsl, ysl, best, path):
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
    hist, xedges, yedges = np.histogram2d(sample_t, sample_a, 
                                          bins=[xbins, ybins])

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
    plt.pcolormesh(xsl[-1] - X, Y, hist, cmap='viridis', vmin=0., vmax=0.1)
    plt.plot(xsl[-1] - xsl, mean, '--r', label="Mean")
    plt.plot(xsl[-1] - xsl, ysl[best, :], '--y', label="Best-Fit")
    for i in df_SL_free.index:
        t_min = df_SL_free.loc[i]['t_min']
        t_max = df_SL_free.loc[i]['t_max']
        e_min = df_SL_free.loc[i]['e_min']
        e_max = df_SL_free.loc[i]['e_max']
        plt.gca().add_patch(Rectangle(
            (t_min, e_min), t_max-t_min, abs(e_max-e_min), 
            linewidth=1, edgecolor='r', facecolor='none')
            )

    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlabel('Age (ka)', fontsize=fs)
    plt.ylabel('Sea-Level (m)', fontsize=fs)
    DefaultSize = fig.get_size_inches()
    fig.set_size_inches((DefaultSize[0] * fig_rescale[0], 
                         DefaultSize[1] * fig_rescale[1]),
                        forward=True)
    plt.tight_layout()
    cbaxes = inset_axes(ax, width="40%", height="5%", loc=1, borderpad=1)
    cbar = plt.colorbar(cax=cbaxes, orientation='horizontal')
    cbar.set_label("Probability", fontsize=fs - 4)
    ax.legend(loc=4)
    plt.savefig(path + "/Sea-Level.pdf")

    fig2 = plt.figure()
    plt.plot(xsl[-1] - xsl, p025, '--g')
    plt.plot(xsl[-1] - xsl, p25, '--b')
    plt.plot(xsl[-1] - xsl, p50, 'r')
    plt.plot(xsl[-1] - xsl, p75, '--b')
    plt.plot(xsl[-1] - xsl, p975, '--g')
    plt.plot(xsl[-1] - xsl, ysl[best, :], '--y')
    plt.xlabel('Age (ka)', fontsize=fs)
    plt.ylabel('Sea-level (m)', fontsize=fs)
    plt.tight_layout()
    ax = plt.gca()
    plt.legend([2.5, 25, 50, 75, 97.5, 'Best-Fit'])
    plt.savefig(path + "/Sea-Level_median_percentiles.pdf")

    return fig, fig2

