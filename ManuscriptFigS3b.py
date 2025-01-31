import os
import sys

# sys.path.append("/home/nhedjazi/src/sealevel")
# sys.path.append("/home/nhedjazi/src/sealevel")
sys.path.insert(0, os.path.abspath('./reef'))
sys.path.insert(1, os.path.abspath('./mcmc'))

# Imports
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import interpolate
from scipy.interpolate import interp1d
from reef import tools as tools
from reef import main as main
from rouzo import Fig2c as Fig2c
from mcmc import misfit as mis
from mcmc.metropolis import Metropolis1dStep
from scipy import linalg
from mcmc import covariance_matrix as cov

# Number of samples, tuner and step-sizes
n_samples = 1000000
n_tune = 1000000
tune_interval = 50000
prop_S = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 2, 2, 0.1])

# load nodes for SL-curve and observed topography
t, e = tools.readfile("SL/Nodes153.dat")
x_obs, y_obs = tools.readfile("../reef/examples/TopoObs153_Fig2c.dat")
tstart = 153  # Length of SL curve
stp = 0  # Starting point for plotting

# REEF parameters
zmax = 20.
zmin = 1
zow = 2
tmax = -1
# zb = 9
# v0 = 200
# u0 = 1.4
G0 = 0
# slopi = 8
dbg = False
dt0 = 0
dx_reef = 10

# Select starting point and bounds
# a0 = 93 + np.random.rand() * 14
# b0 = -28 + np.random.rand() * 20
# a0 = 25. #Midbox value
# b0 = -50. #Midbox value
# a1 = 50. #Midbox value
# b1 = -50. #Midbox value
# a2 = 60. #Midbox value
# b2 = -50. #Midbox value
# a3 = 80. #Midbox value
# b3 = -50. #Midbox value
# a4 = 90. #Midbox value
# b4 = -50. #Midbox value
# a5 = 100. #Midbox value
# b5 = -50. #Midbox value
# a6 = 110. #Midbox value
# b6 = -50. #Midbox value
# a0 = 25. #Mid value
# b0 = -67. #Mid value
# a1 = 49. #Mid value
# b1 = -57. #Mid value
# a2 = 60. #Mid value
# b2 = -68. #Mid value
# a3 = 80. #Mid value
# b3 = -38. #Mid value
# a4 = 90. #Mid value
# b4 = -48. #Mid value
# a5 = 100. #Mid value
# b5 = -34. #Mid value
# a6 = 110. #Mid value
# b6 = -49. #Mid value
a0 = 24. #True value
b0 = -85. #True value
a1 = 47. #True value
b1 = -65. #True value
a2 = 61. #True value
b2 = -87. #True value
a3 = 80. #True value
b3 = -26. #True value
a4 = 90. #True value
b4 = -45. #True value
a5 = 100. #True value
b5 = -18. #True value
a6 = 111. #True value
b6 = -47. #True value
c1 = 200. #Erosion rate
d1 = 8. #Initial slope
e1 = 9. #Wave base
f1 = 1.4 #Uplift Rate
x0 = np.array([a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, c1, d1, e1, f1])
bounds = np.array([[21, 30],
                    [-100, 0],
                    [45, 54],
                    [-100, 0],
                    [55, 64],
                    [-100, 0],
                    [75, 84],
                    [-100, 0],
                    [85, 94],
                    [-100, 0],
                    [95, 104],
                    [-100, 0],
                    [105, 114],
                    [-100, 0],
                    [100, 500],
                    [5, 11],
                    [6, 12],
                    [1.1, 1.7]])

# Vertical interpolation of x_obs and y_obs
ipmin = 0
# ipmax = 100.05
ipmax = 175.05
ipstep = 1
ipobs = interp1d(y_obs, x_obs) # interpolate x_obs as a function of y_obs
y_obs_n = np.arange(ipmin, ipmax, ipstep)
x_obs_n = ipobs(y_obs_n)
x_obs_n = x_obs_n - x_obs_n[0]

# Computing inverse covariance
# n1 = 101    # Number of matrix elements: (ipmax-ipmin) / ipstep, round up to full number
n1 = 176
sigma = 50  # Amplitude in m
corr_l = 10  # Correlation length in m
dx_cov = dx_reef  # Define the length between two samples (in meters)
gamma = 1  # Exponent in the kernel. 1 for laplacian, 2 for gaussian

# First matrix: the physical model error
covar = cov.exponential_covar_1d(
    n1, sigma, corr_l, dx=dx_cov, gamma=gamma, truncate=None)  # Define the covariance
# Second one, measurement error
#eps = 1e-1*sigma*np.identity(n1)
#icovar = linalg.inv(covar+eps)  # This computes the inverse covariance
icovar = linalg.inv(covar)  # This computes the inverse covariance

# Create noise
mean = np.zeros(n1)
#noise1 = np.random.multivariate_normal(mean, covar + eps, 1).flatten()
noise1 = np.random.multivariate_normal(mean, covar, 1).flatten()

# Make noise same length as y_obs and add to y_obs
#nox = np.arange(0, len(y_obs), el)
#npc = interpolate.PchipInterpolator(nox, noise1, axis=0, extrapolate=None)
#noy = np.arange(0, len(y_obs), 1)
#noiseIP = npc(noy)
#y_obs = y_obs + noiseIP

# The proposal
def proposal(x, std):
    assert (x.size == std.size)
    xp = np.copy(x)
    # get the size of vector x
    n = x.size
    # Chose a random element in vector x
    i = np.random.randint(0, n)
    # Add a small perturbation
    dx = np.random.normal(0., std[i])
    xp[i] += dx
    return xp

# The log prior
# The prior 0 outside the bounds. Inside the bounds, it is 1/(domain size).

def prior(x):
    logprior = 0
    n_params = len(x)
    for i in range(n_params):
        if x[i] < bounds[i, 0] or x[i] > bounds[i, 1]:
            logprior = -np.inf  # log(0)
    else:
        # This is optional because it does not depend on x value, I put it for clarity
        for i in range(x.size):
            logprior += -np.log(bounds[i, 1] - bounds[i, 0])
    return logprior

# The loglikelihood
# In future applications, the loglikelihood is the negative of the misfit.

def param(x):
    t[2] = x[0]
    e[2] = x[1]
    t[3] = x[2]
    e[3] = x[3]
    t[4] = x[4]
    e[4] = x[5]
    t[5] = x[6]
    e[5] = x[7]
    t[6] = x[8]
    e[6] = x[9]
    t[7] = x[10]
    e[7] = x[11]
    t[8] = x[12]
    e[8] = x[13]
    v0 = x[14]
    slopi = x[15]
    zb = x[16]
    u0 = x[17]

    pc = interpolate.PchipInterpolator(t, e, axis=0, extrapolate=None)
    tnew = np.arange(152, -1, -1)
    enew = pc(tnew)
    return tnew, enew, v0, slopi, zb, u0

def misfit(x_n, x_obs_n):
    fit = mis.sqmahalanobis(x_n, x_obs_n, icovar)
    return fit

def align(x, y):
    ipmod = interp1d(y, x) # interpolate x as a function of y
    y_n = np.arange(ipmin, ipmax, ipstep)
    x_n = ipmod(y_n)
    x_n = x_n - x_n[0]
    return y_n, x_n

def loglike(x):
    t, e, v0, slopi, zb, u0 = param(x)
    xmin = 100000  # Limits for final plots
    xmax = 0
    ymin = 10000
    ymax = -1000
    x, y, dz, xmin, xmax, ymin, ymax, x_ini, y_ini, dt_final = \
        main.reef(t, e, tmax, u0, slopi, v0, zb, dx_reef, dt0, xmin, xmax,
                      ymin, ymax)
    y_n, x_n = align(x, y)
    fit = misfit(x_n, x_obs_n)
    predictions = {}
    predictions["x_n"] = x_n
    predictions["y_n"] = y_n
    predictions["t"] = t
    predictions["e"] = e
    return -0.5 * fit, predictions
    # return 0, predictions

chain = Metropolis1dStep()
chain.proposal = proposal
chain.logprior = prior
chain.loglikelihood = loglike
chain.show_stats = 10000
chain.prop_S = prop_S

chain.verbose = 0

# Record some statistics
chain.add_stat("loglikelihood")
chain.add_stat("prop_S")
chain.add_stat("accept_ratio")
chain.add_stat("parameter_accept_ratio")

# Run the algorithm
chain.run(x0, n_samples, tune=n_tune, tune_interval=tune_interval,
          discard_tuned_samples=False, thin=1)

print("Total duration:")
print(chain.duration)

# chain.posterior_predictive (check notebook), use a[0,:,:] for arviz
# Some trace plots
fig=plt.figure()
plt.plot(chain.stats["loglikelihood"])
# fig.savefig('Figs/Stats-Loglikelihood.png')
fig.savefig("Figs/FigS3b/Stats-Loglikelihood.pdf", format="pdf", bbox_inches="tight")

fig=plt.figure()
plt.plot(chain.stats["prop_S"][1:])
# fig.savefig('Figs/Stats-prop_S.png')
fig.savefig("Figs/FigS3b/Stats-prop_S.pdf", format="pdf", bbox_inches="tight")

fig=plt.figure()
plt.plot(chain.stats["accept_ratio"][1:])
# fig.savefig('Figs/Stats-accept_ratio.png')
fig.savefig("Figs/FigS3b/Stats-accept_ratio.pdf", format="pdf", bbox_inches="tight")

fig=plt.figure()
plt.plot(chain.stats["parameter_accept_ratio"][1:])
# fig.savefig('Figs/Stats-parameter_accept_ratio.png')
fig.savefig("Figs/FigS3b/Stats-parameter_accept_ratio.pdf", format="pdf", bbox_inches="tight")

# Box 1
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 0][stp:],
                    "SL Elevation (m)" : chain.samples[:, 1][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
                    #marginal_kws=dict(bins=25)
# fig.savefig('Figs/Histogram-2D-1.png')
fig.savefig("Figs/FigS3b/Histogram-2D-1.pdf", format="pdf", bbox_inches="tight")

# Box 2
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 2][stp:],
                    "SL Elevation (m)" : chain.samples[:, 3][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
                    #marginal_kws=dict(bins=25)
# fig.savefig('Figs/Histogram-2D-2.png')
fig.savefig("Figs/FigS3b/Histogram-2D-2.pdf", format="pdf", bbox_inches="tight")

# Box 3
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 4][stp:],
                    "SL Elevation (m)" : chain.samples[:, 5][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
                    #marginal_kws=dict(bins=25)
# fig.savefig('Figs/Histogram-2D-3.png')
fig.savefig("Figs/FigS3b/Histogram-2D-3.pdf", format="pdf", bbox_inches="tight")

# Box 4
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 6][stp:],
                    "SL Elevation (m)" : chain.samples[:, 7][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
                    #marginal_kws=dict(bins=25)
# fig.savefig('Figs/Histogram-2D-4.png')
fig.savefig("Figs/FigS3b/Histogram-2D-4.pdf", format="pdf", bbox_inches="tight")

# Box 5
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 8][stp:],
                    "SL Elevation (m)" : chain.samples[:, 9][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
                    #marginal_kws=dict(bins=25)
# fig.savefig('Figs/Histogram-2D-5.png')
fig.savefig("Figs/FigS3b/Histogram-2D-5.pdf", format="pdf", bbox_inches="tight")

# Box 6
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 10][stp:],
                    "SL Elevation (m)" : chain.samples[:, 11][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
# fig.savefig('Figs/Histogram-2D-6.png')
fig.savefig("Figs/FigS3b/Histogram-2D-6.pdf", format="pdf", bbox_inches="tight")

# Box 7
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 12][stp:],
                    "SL Elevation (m)" : chain.samples[:, 13][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
# fig.savefig('Figs/Histogram-2D-7.png')
fig.savefig("Figs/FigS3b/Histogram-2D-7.pdf", format="pdf", bbox_inches="tight")

# Box 2.1
df = pd.DataFrame({"Erosion rate (mm/yr)" : chain.samples[:, 14][stp:],
                    "Initial slope (%)" : chain.samples[:, 15][stp:]})
fig = sns.jointplot(data=df, x="Erosion rate (mm/yr)", y="Initial slope (%)", kind="hex", palette="colorblind")
fig.savefig("Figs/FigS3b/Histogram-2D-ER-IS.pdf", format="pdf", bbox_inches="tight")

# Box 2.2
df = pd.DataFrame({"Erosion rate (mm/yr)" : chain.samples[:, 14][stp:],
                    "Wave base depth (m)" : chain.samples[:, 16][stp:]})
fig = sns.jointplot(data=df, x="Erosion rate (mm/yr)", y="Wave base depth (m)", kind="hex", palette="colorblind")
fig.savefig("Figs/FigS3b/Histogram-2D-ER-WB.pdf", format="pdf", bbox_inches="tight")

# Box 2.3
df = pd.DataFrame({"Initial slope (%)" : chain.samples[:, 15][stp:],
                    "Wave base depth (m)" : chain.samples[:, 16][stp:]})
fig = sns.jointplot(data=df, x="Initial slope (%)", y="Wave base depth (m)", kind="hex", palette="colorblind")
fig.savefig("Figs/FigS3b/Histogram-2D-IS-WB.pdf", format="pdf", bbox_inches="tight")

# Box 2.4
df = pd.DataFrame({"Erosion rate (mm/yr)" : chain.samples[:, 14][stp:],
                    "Uplift Rate (mm/yr)" : chain.samples[:, 17][stp:]})
fig = sns.jointplot(data=df, x="Erosion rate (mm/yr)", y="Uplift Rate (mm/yr)", kind="hex", palette="colorblind")
fig.savefig("Figs/FigS3b/Histogram-2D-ER-UR.pdf", format="pdf", bbox_inches="tight")

# Box 2.5
df = pd.DataFrame({"Initial slope (%)" : chain.samples[:, 15][stp:],
                    "Uplift Rate (mm/yr)" : chain.samples[:, 17][stp:]})
fig = sns.jointplot(data=df, x="Initial slope (%)", y="Uplift Rate (mm/yr)", kind="hex", palette="colorblind")
fig.savefig("Figs/FigS3b/Histogram-2D-IS-UR.pdf", format="pdf", bbox_inches="tight")

# Box 2.6
df = pd.DataFrame({"Wave base depth (m)" : chain.samples[:, 16][stp:],
                    "Uplift Rate (mm/yr)" : chain.samples[:, 17][stp:]})
fig = sns.jointplot(data=df, x="Wave base depth (m)", y="Uplift Rate (mm/yr)", kind="hex", palette="colorblind")
fig.savefig("Figs/FigS3b/Histogram-2D-WB-UR.pdf", format="pdf", bbox_inches="tight")

# Profile plot
best = np.argmax(chain.stats["loglikelihood"][stp:])
x_n = chain.posterior_predictive["x_n"][0, stp:, :]
y_n = chain.posterior_predictive["y_n"][0, 0, :]
fig, fig2 = Fig2c.profile(x_n, y_n, x_obs_n, y_obs_n, best)

# Sea-level plot
xsl = np.arange(0, tstart, 1)
ysl = chain.posterior_predictive["e"][0, :, :][stp:]
fig, fig2 = Fig2c.sealevel(xsl, ysl, best)

plt.show()

# If you don't use arviz for plots but only for saving the data, I have created this command:
# chain.write_samples("test_results_arviz.nc", format='arviz')
# import arviz
# Transform results to a arviz dataset
# dataset = chain.get_results(format='arviz')

# Test read/write to netCDF format
# Save it
# dataset.to_netcdf("test_results_arviz.nc")
