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
from rouzo import FigS2 as FigS2
from mcmc import misfit as mis
from mcmc.metropolis import Metropolis1dStep
from scipy import linalg
from mcmc import covariance_matrix as cov

# Number of samples, tuner and step-sizes
n_samples = 10000
n_tune = 1000000
tune_interval = 50000
prop_S = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

# load nodes for SL-curve and observed topography
t, e = tools.readfile("SL/Nodes80.dat")
x_obs, y_obs = tools.readfile("../reef/examples/TopoObs_Fig2a.dat")
tstart = 80  # Length of SL curve
stp = 0  # Starting point for plotting

# REEF parameters
zmax = 20.
zmin = 1
zow = 2
tmax = -1
zb = 12
v0 = 200
u0 = 2
G0 = 0
slopi = 5
dbg = False
dt0 = 0
dx_reef = 1

# Select starting point and bounds
# a0 = 93 + np.random.rand() * 14
# b0 = -28 + np.random.rand() * 20
# a0 = 43. #Mid value
# b0 = -70. #Mid value
# a1 = 22. #Mid value
# b1 = -70. #Mid value
# a2 = 64. #Mid value
# b2 = -70. #Mid value
a0 = 46. #True value
b0 = -58. #True value
a1 = 24. #True value
b1 = -130. #True value
a2 = 61. #True value
b2 = -87. #True value
# a0 = 47. #NearTrue value
# b0 = -57. #NearTrue value
# a1 = 23. #NearTrue value
# b1 = -129. #NearTrue value
# a2 = 62. #NearTrue value
# b2 = -88. #NearTrue value
x0 = np.array([a0, b0, a1, b1, a2, b2])
bounds = np.array([[33, 53],
                    [-140, 0],
                    [12, 32],
                    [-140, 0],
                    [54, 74],
                    [-140, 0]])

# Vertical interpolation of x_obs and y_obs
ipmin = 1015
ipmax = 2215
ipstep = 150
ipobs = interp1d(x_obs, y_obs) # interpolate x_obs as a function of y_obs
x_obs_n = np.arange(ipmin, ipmax, ipstep)
y_obs_n = ipobs(x_obs_n)
x_obs_n = x_obs_n - x_obs_n[0]

# Computing inverse covariance
n1 = 8    # Number of matrix elements: (ipmax-ipmin) / ipstep, round up to full number
sigma = 1  # Amplitude in m
corr_l = 1  # Correlation length in m
dx_cov = dx_reef  # Define the length between two samples (in meters)
gamma = 1  # Exponent in the kernel. 1 for laplacian, 2 for gaussian

# ### Topo obs
# ipmin = min(x_obs)  # Min x value from observed topo
# ipmax = max(x_obs)  # Max x value from observed topo
# # y_obs_min = min(y_obs)
# # y_obs_max = max(y_obs)
# y_obs_min = 0
# y_obs_max = 100
#
# ### Interpolation values
# ipstep = 50
# sigma = 2  # Amplitude in m
# corr_l = 3  # Correlation length in m
#
# ## Horizontal interpolation of x_obs and y_obs
# ipobs = interp1d(x_obs, y_obs)  # interpolate y_obs as a function of x_obs
# x_obs_n = np.arange(ipmin, ipmax, ipstep)
# y_obs_n = ipobs(x_obs_n)
#
# ## Computing inverse covariance
# n1 = int(-(-(ipmax-ipmin)//ipstep)) # Number of matrix elements, round up to full number
# dx_cov=dx_reef
# gamma = 1  # Exponent in the kernel. 1 for laplacian, 2 for gaussian

# First matrix: the physical model error
covar = cov.exponential_covar_1d(
    n1, sigma, corr_l, dx=dx_cov, gamma=gamma, truncate=None)  # Define the covariance

# Second one, measurement error
icovar = linalg.inv(covar)  # This computes the inverse covariance

# Create noise
mean = np.zeros(n1)
noise1 = np.random.multivariate_normal(mean, covar, 1).flatten()

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
    t[3] = x[0]
    e[3] = x[1]
    t[2] = x[2]
    e[2] = x[3]
    t[4] = x[4]
    e[4] = x[5]
    pc = interpolate.PchipInterpolator(t, e, axis=0, extrapolate=None)
    tnew = np.arange(79, -1, -1)
    enew = pc(tnew)
    return tnew, enew

def misfit(y_n, y_obs_n):
    "Check the misfit between z value of the simulation and z value of topo_obs"
    fit = mis.sqmahalanobis(y_n, y_obs_n, icovar)
    return fit

# def align(x, y):
#     "Cut the simulated array to plot only the area of interest and compare with topo_obs"
#     index_min = np.argmax(y >= y_obs_min)
#     x_start = x[index_min]
#
#     ipmod = interp1d(x, y)  # interpolate y as a function of x
#     x_n = np.arange(x_start, x_start + ipmin+ipmax, ipstep)
#     y_n = ipmod(x_n)
#     x_n = x_n - x_n[0]
#     return y_n, x_n

# def align(x, y):
#     ipmod = interp1d(x, y) # interpolate x as a function of y
#     x_n = np.arange(ipmin, ipmax, ipstep)
#     y_n = ipmod(x_n)
#     x_n = x_n - x_n[0]
#     return y_n, x_n


def align(x, y):
    "Cut the simulated array to plot only the area of interest and compare with topo_obs"
    y_obs_min = 0
    index_min = np.argmax(y >= y_obs_min)
    x_start = x[index_min]

    ipmod = interp1d(x, y)  # interpolate y as a function of x
    x_n = np.arange(x_start, x_start - ipmin + ipmax, ipstep)
    y_n = ipmod(x_n)
    x_n = x_n - x_n[0]
    x_p = np.arange(x_start, x_start - ipmin + ipmax, 5)
    y_p = ipmod(x_p)
    x_p = x_p - x_p[0]
    return y_n, x_n, y_p, x_p

def loglike(x):
    t, e = param(x)
    xmin = 100000  # Limits for final plots
    xmax = -100000
    ymin = 100000
    ymax = -100000
    x, y, dz, xmin, xmax, ymin, ymax, x_ini, y_ini, dt_final = \
        main.reef(t, e, tmax, u0, slopi, v0, zb, dx_reef, dt0, xmin, xmax,
                      ymin, ymax)
    y_n, x_n, y_p, x_p = align(x, y)
    fit = misfit(y_n, y_obs_n)
    predictions = {}
    predictions["x_n"] = x_n
    predictions["y_n"] = y_n
    predictions["x_p"] = x_p
    predictions["y_p"] = y_p
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
plt.plot(chain.stats["loglikelihood"][1:])
plt.savefig("Figs/FigS2/Stats-Loglikelihood.pdf", format="pdf", bbox_inches="tight")

fig=plt.figure()
plt.plot(chain.stats["prop_S"][1:])
plt.savefig("Figs/FigS2/Stats-prop_S.pdf", format="pdf", bbox_inches="tight")

fig=plt.figure()
plt.plot(chain.stats["accept_ratio"][1:])
plt.savefig("Figs/FigS2/Stats-accept_ratio.pdf", format="pdf", bbox_inches="tight")

fig=plt.figure()
plt.plot(chain.stats["parameter_accept_ratio"][1:])
plt.savefig("Figs/FigS2/Stats-parameter_accept_ratio.pdf", format="pdf", bbox_inches="tight")

# Profile plot
best = np.argmax(chain.stats["loglikelihood"][stp:])
x = chain.posterior_predictive["x_p"][0, 0, :]
y = chain.posterior_predictive["y_p"][0, :, :]
fig2 = FigS2.profile(x, y, x_obs, y_obs, best)

mean = np.mean(y, axis=0)
median = np.percentile(y[:, :], 50, axis=0)
best_prof = y[best, :]
np.savetxt("Figs/FigS2/MeanProfile.txt", mean)
np.savetxt("Figs/FigS2/MedianProfile.txt", median)
np.savetxt("Figs/FigS2/BestProfile.txt", best_prof)

all_loglikes = chain.stats["loglikelihood"][stp:, :]
best_loglike = all_loglikes[best, :]
np.savetxt("Figs/FigS2/BestLogLike.txt", best_loglike)

# Sea-level plot
xsl = np.arange(0, tstart, 1)
ysl = chain.posterior_predictive["e"][0, :, :][stp:]
fig, fig2 = FigS2.sealevel(xsl, ysl, best)

mean = np.mean(ysl, axis=0)
median = np.percentile(ysl[:, :], 50, axis=0)
best_sl = ysl[best, :]
np.savetxt("Figs/FigS2/MeanSL.txt", mean)
np.savetxt("Figs/FigS2/MedianSL.txt", median)
np.savetxt("Figs/FigS2/BestSL.txt", best_sl)

# Box 1.1
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 0][stp:],
                    "SL Elevation (m)" : chain.samples[:, 1][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
plt.savefig("Figs/FigS2/Histogram-2D-1.pdf", format="pdf", bbox_inches="tight")

# Box 1.2
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 2][stp:],
                    "SL Elevation (m)" : chain.samples[:, 3][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
plt.savefig("Figs/FigS2/Histogram-2D-2.pdf", format="pdf", bbox_inches="tight")

# Box 1.3
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 4][stp:],
                    "SL Elevation (m)" : chain.samples[:, 5][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
plt.savefig("Figs/FigS2/Histogram-2D-3.pdf", format="pdf", bbox_inches="tight")

# Parameter path 1
fig=plt.figure()
plt.plot(chain.samples[:, 0], chain.samples[:, 1])
plt.xlabel('Age (ka)')
plt.ylabel('SL Elevation (m)')
plt.xlim([33, 53])
plt.ylim([-140, 0])
plt.savefig("Figs/FigS2/Path1.pdf", format="pdf", bbox_inches="tight")

# Parameter path 2
fig=plt.figure()
plt.plot(chain.samples[:, 2], chain.samples[:, 3])
plt.xlabel('Age (ka)')
plt.ylabel('SL Elevation (m)')
plt.xlim([12, 32])
plt.ylim([-140, 0])
plt.savefig("Figs/FigS2/Path2.pdf", format="pdf", bbox_inches="tight")

# Parameter path 3
fig=plt.figure()
plt.plot(chain.samples[:, 4], chain.samples[:, 5])
plt.xlabel('Age (ka)')
plt.ylabel('SL Elevation (m)')
plt.xlim([54, 74])
plt.ylim([-140, 0])
plt.savefig("Figs/FigS2/Path3.pdf", format="pdf", bbox_inches="tight")

plt.show()
