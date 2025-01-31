import os
import sys

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
from rouzo import Fig2d as Fig2d
from mcmc import misfit as mis
from mcmc.metropolis import Metropolis1dStep
from scipy import linalg
from mcmc import covariance_matrix as cov

# Number of samples, tuner and step-sizes
n_samples = 1000000
n_tune = 1000000
tune_interval = 50000
prop_S = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])

# load nodes for SL-curve and observed topography
n_topo = 2  # number of profiles to consider
t, e = tools.readfile("SL/Nodes153.dat")
x_obs = []
y_obs = []
for i in range(n_topo):
     x_obs_tmp, y_obs_tmp = tools.readfile("../reef/examples/TopoObs153_{}.dat".format(i))
     x_obs.append(x_obs_tmp)
     y_obs.append(y_obs_tmp)
tstart = 153  # Length of SL curve
stp = 0  # Starting point for plotting

# REEF parameters
zmax = 20.
zmin = 1
zow = 2
tmax = -1
zb = 9
v0 = 200
u0 = [0.7, 1.4]
G0 = 0
slopi = 8
dbg = False
dt0 = 0
dx_reef = 1

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
x0 = np.array([a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6])
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
                    [-100, 0]])

# Vertical interpolation of x_obs and y_obs
ipmin = [0, 0]
ipmax = [100.05, 175.05]
ipstep = 1
for i in range(n_topo):
     ipobs = interp1d(y_obs[i], x_obs[i]) # interpolate x_obs as a function of y_obs
     y_obs[i] = np.arange(ipmin[i], ipmax[i], ipstep)
     x_obs[i] = ipobs(y_obs[i])
     x_obs[i] = x_obs[i] - x_obs[i][0]

# Computing inverse covariance
n1 = [101, 176]    # Number of matrix elements: (ipmax-ipmin) / ipstep, round up to full number
sigma = 50  # Amplitude in m
corr_l = 10  # Correlation length in m
dx_cov = dx_reef  # Define the length between two samples (in meters)
gamma = 1  # Exponent in the kernel. 1 for laplacian, 2 for gaussian

# First matrix: the physical model error
icovar = []
for i in range(n_topo):
    covar = cov.exponential_covar_1d(
        n1[i], sigma, corr_l, dx=dx_cov, gamma=gamma, truncate=None)  # Define the covariance
    # Second one, measurement error
    #eps = 1e-1*sigma*np.identity(n1)
    #icovar = linalg.inv(covar+eps)  # This computes the inverse covariance
    icovar.append(linalg.inv(covar))  # This computes the inverse covariance

# Create noise
#mean = np.zeros(n1)
#noise1 = np.random.multivariate_normal(mean, covar + eps, 1).flatten()
#noise1 = np.random.multivariate_normal(mean, covar, 1).flatten()

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
    pc = interpolate.PchipInterpolator(t, e, axis=0, extrapolate=None)
    tnew = np.arange(152, -1, -1)
    enew = pc(tnew)
    return tnew, enew

def misfit(x_n, x_obs, icovar_i):
    fit = mis.sqmahalanobis(x_n, x_obs, icovar_i)
    return fit

def align(x, y, ipmin, ipmax):
    ipmod = interp1d(y, x) # interpolate x as a function of y
    y_na = np.arange(ipmin, ipmax, ipstep)
    x_na = ipmod(y_na)
    x_na = x_na - x_na[0]
    return y_na, x_na

def loglike(x):
    t, e = param(x)
    xmin = 100000  # Limits for final plots
    xmax = 0
    ymin = 10000
    ymax = -1000

    sum_fit = 0
    predictions = {}
    for i in range(n_topo):
        x, y, dz, xmin, xmax, ymin, ymax, x_ini, y_ini, dt_final = \
            main.reef(t, e, tmax, u0[i], slopi, v0, zb, dx_reef, dt0, xmin, xmax, ymin, ymax)
        y_n, x_n = align(x, y, ipmin[i], ipmax[i])
        fit = misfit(x_n, x_obs[i], icovar[i])
        xchar = "x_{}".format(i)
        ychar = "y_{}".format(i)

        predictions[xchar] = x_n
        predictions[ychar] = y_n
        sum_fit += fit

    predictions["t"] = t
    predictions["e"] = e

    return -0.5 * sum_fit, predictions
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
fig.savefig("Figs/Fig2d/Stats-Loglikelihood.pdf", format="pdf", bbox_inches="tight")

fig=plt.figure()
plt.plot(chain.stats["prop_S"][1:])
# fig.savefig('Figs/Stats-prop_S.png')
fig.savefig("Figs/Fig2d/Stats-prop_S.pdf", format="pdf", bbox_inches="tight")

fig=plt.figure()
plt.plot(chain.stats["accept_ratio"][1:])
# fig.savefig('Figs/Stats-accept_ratio.png')
fig.savefig("Figs/Fig2d/Stats-accept_ratio.pdf", format="pdf", bbox_inches="tight")

fig=plt.figure()
plt.plot(chain.stats["parameter_accept_ratio"][1:])
# fig.savefig('Figs/Stats-parameter_accept_ratio.png')
fig.savefig("Figs/Fig2d/Stats-parameter_accept_ratio.pdf", format="pdf", bbox_inches="tight")

# Box 1
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 0][stp:],
                    "SL Elevation (m)" : chain.samples[:, 1][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
                    #marginal_kws=dict(bins=25)
# fig.savefig('Figs/Histogram-2D-1.png')
fig.savefig("Figs/Fig2d/Histogram-2D-1.pdf", format="pdf", bbox_inches="tight")

# Box 2
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 2][stp:],
                    "SL Elevation (m)" : chain.samples[:, 3][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
                    #marginal_kws=dict(bins=25)
# fig.savefig('Figs/Histogram-2D-2.png')
fig.savefig("Figs/Fig2d/Histogram-2D-2.pdf", format="pdf", bbox_inches="tight")

# Box 3
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 4][stp:],
                    "SL Elevation (m)" : chain.samples[:, 5][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
                    #marginal_kws=dict(bins=25)
# fig.savefig('Figs/Histogram-2D-3.png')
fig.savefig("Figs/Fig2d/Histogram-2D-3.pdf", format="pdf", bbox_inches="tight")

# Box 4
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 6][stp:],
                    "SL Elevation (m)" : chain.samples[:, 7][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
                    #marginal_kws=dict(bins=25)
# fig.savefig('Figs/Histogram-2D-4.png')
fig.savefig("Figs/Fig2d/Histogram-2D-4.pdf", format="pdf", bbox_inches="tight")

# Box 5
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 8][stp:],
                    "SL Elevation (m)" : chain.samples[:, 9][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
                    #marginal_kws=dict(bins=25)
# fig.savefig('Figs/Histogram-2D-5.png')
fig.savefig("Figs/Fig2d/Histogram-2D-5.pdf", format="pdf", bbox_inches="tight")

# Box 6
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 10][stp:],
                    "SL Elevation (m)" : chain.samples[:, 11][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
# fig.savefig('Figs/Histogram-2D-6.png')
fig.savefig("Figs/Fig2d/Histogram-2D-6.pdf", format="pdf", bbox_inches="tight")

# Box 7
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 12][stp:],
                    "SL Elevation (m)" : chain.samples[:, 13][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
# fig.savefig('Figs/Histogram-2D-7.png')
fig.savefig("Figs/Fig2d/Histogram-2D-7.pdf", format="pdf", bbox_inches="tight")

# # Profile plot
best = np.argmax(chain.stats["loglikelihood"][stp:])

x_obs_tmp, y_obs_tmp = tools.readfile("../reef/examples/TopoObs153_{}.dat".format(i))

for i in range(n_topo):
    x_n = chain.posterior_predictive["x_{}".format(i)][0, stp:, :]
    y_n = chain.posterior_predictive["y_{}".format(i)][0, 0, :]
    fig, fig2 = Fig2d.profile(x_n, y_n, x_obs, y_obs, best, i)

# Sea-level plot
xsl = np.arange(0, tstart, 1)
ysl = chain.posterior_predictive["e"][0, :, :][stp:]
fig, fig2 = Fig2d.sealevel(xsl, ysl, best)

plt.show()

# If you don't use arviz for plots but only for saving the data, I have created this command:
# chain.write_samples("test_results_arviz.nc", format='arviz')
# import arviz
# Transform results to a arviz dataset
# dataset = chain.get_results(format='arviz')

# Test read/write to netCDF format
# Save it
# dataset.to_netcdf("test_results_arviz.nc")
