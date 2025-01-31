import os
import sys

# sys.path.append("/home/nhedjazi/src/sealevel")
sys.path.insert(1, os.path.abspath('../'))

# Imports
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import interpolate
from scipy.interpolate import interp1d
from reef import tools as tools
from reef import main as main
from ScriptsFigs_deGelder_etal_2025 import Fig5c as Fig5c
from mcmc import misfit as mis
from mcmc.metropolis import Metropolis1dStep
from scipy import linalg
from mcmc import covariance_matrix as cov

# Number of samples, tuner and step-sizes
n_samples = 1000000
n_tune = 1000000
tune_interval = 50000
prop_S = np.array([1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,
                   1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,
                   1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,
                   1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 1, 25, 0.5, 0.5, 0.1])

# 50, 2, 2, 0.05

# load nodes for SL-curve and observed topography
t, e = tools.readfile("../SL_nodes/Nodes450.dat")
#x_obs, y_obs = tools.readfile("../Topo_obs/TopoObs_Corinth_Prof1.dat")
x_obs, y_obs = tools.readfile("../Topo_obs/TopoObs_Corinth_Prof2.dat")
tstart = 450  # Length of SL curve
stp = 10000  # Starting point for plotting

# REEF parameters
zmax = 20.
zmin = 1
zow = 2
tmax = -1
G0 = 0
dbg = False
dt0 = 0
dx_reef = 10

#SL Curve maximumum rates of change
r_max = 100.
f_max = 100.

# Select starting point and bounds
a0 = 6.
b0 = 0.5
a1 = 23.
b1 = 0.5
a2 = 42.
b2 = 0.5
a3 = 48.
b3 = 0.5
a4 = 53.
b4 = 0.5
a5 = 64.
b5 = 0.5
a6 = 80.
b6 = 0.8
a7 = 90.
b7 = 0.7
a8 = 100.
b8 = 0.5
a9 = 111.
b9 = 0.7
a10 = 123.
b10 = 0.5
a11 = 138.
b11 = 0.2
a12 = 149.
b12 = 0.3
a13 = 159.
b13 = 0.5
a14 = 170.
b14 = 0.7
a15 = 184.
b15 = 0.6
a16 = 197.
b16 = 0.5
a17 = 204.
b17 = 0.1
a18 = 213.
b18 = 0.5
a19 = 225.
b19 = 0.8
a20 = 237.
b20 = 0.3
a21 = 251.
b21 = 0.4
a22 = 261.
b22 = 0.5
a23 = 270.
b23 = 0.4
a24 = 281.
b24 = 0.8
a25 = 295.
b25 = 0.8
a26 = 311.
b26 = 0.9
a27 = 319.
b27 = 0.9
a28 = 327.
b28 = 0.8
a29 = 341.
b29 = 0.4
a30 = 352.
b30 = 0.4
a31 = 359.
b31 = 0.4
a32 = 367.
b32 = 0.8
a33 = 377.
b33 = 0.7
a34 = 384.
b34 = 0.9
a35 = 393.
b35 = 0.9
a36 = 406.
b36 = 0.5
a37 = 433.
b37 = -121.
# c1 = 400.  #Erosion rate Profile 1
# d1 = 10.5    #Initial slope Profile 1
# e1 = 8.5    #Wave base Profile 1
# f1 = 1.3  #Uplift Rate Profile 1
# c1 = 700.    #Erosion rate Profile 2
# d1 = 5.      #Initial slope Profile 2
# e1 = 6.      #Wave base Profile 2
# f1 = 0.75    #Uplift Rate Profile 2
c1 = 300.    #Erosion rate Profile 3
d1 = 4.2      #Initial slope Profile 3
e1 = 5.      #Wave base Profile 3
f1 = 0.53    #Uplift Rate Profile 3

x0 = np.array([a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7, a8, b8, a9, b9,
               a10, b10, a11, b11, a12, b12, a13, b13, a14, b14, a15, b15, a16, b16, a17, b17, a18, b18, a19, b19,
               a20, b20, a21, b21, a22, b22, a23, b23, a24, b24, a25, b25, a26, b26, a27, b27, a28, b28, a29, b29,
               a30, b30, a31, b31, a32, b32, a33, b33, a34, b34, a35, b35, a36, b36, a37, b37, c1, d1, e1, f1])
bounds = np.array([[4, 8],
                    [0, 1],
                    [19, 27],
                    [0, 1],
                    [36, 43],
                    [0, 1],
                    [44, 49],
                    [0, 1],
                    [50, 57],
                    [0, 1],
                    [58, 74],
                    [0, 1],
                    [75, 85],
                    [0, 1],
                    [86, 94],
                    [0, 1],
                    [95, 105],
                    [0, 1],
                    [106, 117],
                    [0, 1],
                    [118, 128],
                    [0, 1],
                    [132, 143],
                    [0, 1],
                    [144, 155],
                    [0, 1],
                    [156, 163],
                    [0, 1],
                    [164, 177],
                    [0, 1],
                    [178, 191],
                    [0, 1],
                    [192, 201],
                    [0, 1],
                    [202, 207],
                    [0, 1],
                    [208, 219],
                    [0, 1],
                    [220, 231],
                    [0, 1],
                    [232, 243],
                    [0, 1],
                    [245, 256],
                    [0, 1],
                    [257, 265],
                    [0, 1],
                    [266, 275],
                    [0, 1],
                    [276, 286],
                    [0, 1],
                    [287, 304],
                    [0, 1],
                    [305, 317],
                    [0, 1],
                    [318, 321],
                    [0, 1],
                    [322, 332],
                    [0, 1],
                    [335, 348],
                    [0, 1],
                    [349, 354],
                    [0, 1],
                    [355, 362],
                    [0, 1],
                    [363, 372],
                    [0, 1],
                    [373, 380],
                    [0, 1],
                    [381, 388],
                    [0, 1],
                    [389, 397],
                    [0, 1],
                    [398, 414],
                    [0, 1],
                    [426, 438],
                    [-150, -85],
                    [100, 1500],
                    [1, 20],
                    [1, 12],
                    [0.4, 0.55]])

#1.25, 1.4 Profile 1
#0.7, 0.9 Profile 2
#0.4, 0.55 Profile 3

# Vertical interpolation of x_obs and y_obs
ipmin = 40.
ipmax = 140.05
# ipmin = -44, 200, 40
# ipmax = 375.05, 350.05, 140.05
ipstep = 1  # related to uncertainty in terrace height
ipobs = interp1d(y_obs, x_obs)  # interpolate x_obs as a function of y_obs
y_obs_n = np.arange(ipmin, ipmax, ipstep)
x_obs_n = ipobs(y_obs_n)
x_obs_n = x_obs_n - x_obs_n[0]

# Computing inverse covariance
n1 = 101    # Number of matrix elements: (ipmax-ipmin) / ipstep, round up to full number
# n1 = 420, 351, 101    # Number of matrix elements: (ipmax-ipmin) / ipstep, round up to full number
sigma = 50  # Amplitude in m, related to uncertainty in terrace width
# sigma = 50, 100, 25  # Amplitude in m, related to uncertainty in terrace width
corr_l = 10  # Correlation length in m, related to uncertainty in cliff height
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
    t[1] = x[0]
    t[2] = x[2]
    t[3] = x[4]
    t[4] = x[6]
    t[5] = x[8]
    t[6] = x[10]
    t[7] = x[12]
    t[8] = x[14]
    t[9] = x[16]
    t[10] = x[18]
    t[11] = x[20]
    t[12] = x[22]
    t[13] = x[24]
    t[14] = x[26]
    t[15] = x[28]
    t[16] = x[30]
    t[17] = x[32]
    t[18] = x[34]
    t[19] = x[36]
    t[20] = x[38]
    t[21] = x[40]
    t[22] = x[42]
    t[23] = x[44]
    t[24] = x[46]
    t[25] = x[48]
    t[26] = x[50]
    t[27] = x[52]
    t[28] = x[54]
    t[29] = x[56]
    t[30] = x[58]
    t[31] = x[60]
    t[32] = x[62]
    t[33] = x[64]
    t[34] = x[66]
    t[35] = x[68]
    t[36] = x[70]
    t[37] = x[72]
    t[38] = x[74]
    e[38] = x[75]

    e_max = [13, 9, 9, -15, -15, -15, -15, -15, 15, 9, 11, -15, -15, -15, -15, -15, 11, -15, 11, 5, 5, -15,
             -15, -15, -15, -15, 9, -15, -5, -15, -15, -15, -15, -15, -15, -50, 6]
    e_min = [-6, -150, -150, -150, -150, -150, -150, -150, -30, -150, -150, -150, -150, -150, -150, -150, -36,
             -150, -31, -38, -30, -150, -150, -150, -150, -150, 0, -150, -36, -150, -150, -150, -150, -150,
             -150, -150, 0]

    for i in range(37):
        if ((t[38-i] - t[37-i]) * r_max) + e[38-i] < e_max[i]:
            e_max[i] = ((t[38-i] - t[37-i]) * r_max) + e[38-i]
        if e[38-i] - ((t[38-i] - t[37-i]) * f_max) > e_min[i]:
            e_min[i] = e[38-i] - ((t[38-i] - t[37-i]) * f_max)
        e[37-i] = e_min[i] + (x[73-(2*i)] * (e_max[i] - e_min[i]))

    v0 = x[76]
    slopi = x[77]
    zb = x[78]
    u0 = x[79]
    pc = interpolate.PchipInterpolator(t, e, axis=0, extrapolate=None)
    tnew = np.arange(tstart-1, -1, -1)
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
plt.plot(chain.stats["loglikelihood"][1:])
fig.savefig('../Figs/Fig5c/Stats-Loglikelihood.pdf')

fig=plt.figure()
plt.plot(chain.stats["prop_S"][1:])
fig.savefig('../Figs/Fig5c/Stats-prop_S.pdf')

fig=plt.figure()
plt.plot(chain.stats["accept_ratio"][1:])
fig.savefig('../Figs/Fig5c/Stats-accept_ratio.pdf')

fig=plt.figure()
plt.plot(chain.stats["parameter_accept_ratio"][1:])
fig.savefig('../Figs/Fig5c/Stats-parameter_accept_ratio.pdf')


# Profile plot
best = np.argmax(chain.stats["loglikelihood"][stp:])
x_n = chain.posterior_predictive["x_n"][0, stp:, :]
y_n = chain.posterior_predictive["y_n"][0, 0, :]
fig, fig2 = Fig5c.profile(x_n, y_n, x_obs_n, y_obs_n, best)

all_loglikes = chain.stats["loglikelihood"][stp:, :]
best_loglike = all_loglikes[best, :]
np.savetxt("../Figs/Fig5c/BestLogLike.txt", best_loglike)

# Sea-level plot
xsl = np.arange(0, tstart, 1)
ysl = chain.posterior_predictive["e"][0, :, :][stp:]
fig, fig2 = Fig5c.sealevel(xsl, ysl, best)

mean = np.mean(ysl, axis=0)
median = np.percentile(ysl[:, :], 50, axis=0)
best_sl = ysl[best, :]
np.savetxt("../Figs/Fig5c/MeanSL.txt", mean)
np.savetxt("../Figs/Fig5c/MedianSL.txt", median)
np.savetxt("../Figs/Fig5c/BestSL.txt", best_sl)

# Box 1.1
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 0][stp:],
                    "SL Elevation fraction" : chain.samples[:, 1][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-1.pdf')

# Box 1.2
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 2][stp:],
                    "SL Elevation fraction" : chain.samples[:, 3][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-2.pdf')

# Box 1.3
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 4][stp:],
                    "SL Elevation fraction" : chain.samples[:, 5][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-3.pdf')

# Box 1.4
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 6][stp:],
                    "SL Elevation fraction" : chain.samples[:, 7][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-4.pdf')

# Box 1.5
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 8][stp:],
                    "SL Elevation fraction" : chain.samples[:, 9][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-5.pdf')

# Box 1.6
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 10][stp:],
                    "SL Elevation fraction" : chain.samples[:, 11][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-6.pdf')

# Box 1.7
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 12][stp:],
                    "SL Elevation fraction" : chain.samples[:, 13][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-7.pdf')

# Box 1.8
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 14][stp:],
                    "SL Elevation fraction" : chain.samples[:, 15][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-8.pdf')

# Box 1.9
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 16][stp:],
                    "SL Elevation fraction" : chain.samples[:, 17][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-9.pdf')

# Box 1.10
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 18][stp:],
                    "SL Elevation fraction" : chain.samples[:, 19][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-10.pdf')

# Box 1.11
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 20][stp:],
                    "SL Elevation fraction" : chain.samples[:, 21][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-11.pdf')

# Box 1.12
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 22][stp:],
                    "SL Elevation fraction" : chain.samples[:, 23][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-12.pdf')

# Box 1.13
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 24][stp:],
                    "SL Elevation fraction" : chain.samples[:, 25][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-13.pdf')

# Box 1.14
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 26][stp:],
                    "SL Elevation fraction" : chain.samples[:, 27][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-14.pdf')

# Box 1.15
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 28][stp:],
                    "SL Elevation fraction" : chain.samples[:, 29][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-15.pdf')

# Box 1.16
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 30][stp:],
                    "SL Elevation fraction" : chain.samples[:, 31][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-16.pdf')

# Box 1.17
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 32][stp:],
                    "SL Elevation fraction" : chain.samples[:, 33][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-17.pdf')

# Box 1.18
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 34][stp:],
                    "SL Elevation fraction" : chain.samples[:, 35][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-18.pdf')

# Box 1.19
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 36][stp:],
                    "SL Elevation fraction" : chain.samples[:, 37][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-19.pdf')

# Box 1.20
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 38][stp:],
                    "SL Elevation fraction" : chain.samples[:, 39][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-20.pdf')

# Box 1.21
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 40][stp:],
                    "SL Elevation fraction" : chain.samples[:, 41][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-21.pdf')

# Box 1.22
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 42][stp:],
                    "SL Elevation fraction" : chain.samples[:, 43][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-22.pdf')

# Box 1.23
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 44][stp:],
                    "SL Elevation fraction" : chain.samples[:, 45][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-23.pdf')

# Box 1.24
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 46][stp:],
                    "SL Elevation fraction" : chain.samples[:, 47][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-24.pdf')

# Box 1.25
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 48][stp:],
                    "SL Elevation fraction" : chain.samples[:, 49][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-25.pdf')

# Box 1.26
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 50][stp:],
                    "SL Elevation fraction" : chain.samples[:, 51][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-26.pdf')

# Box 1.27
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 52][stp:],
                    "SL Elevation fraction" : chain.samples[:, 53][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-27.pdf')

# Box 1.28
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 54][stp:],
                    "SL Elevation fraction" : chain.samples[:, 55][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-28.pdf')

# Box 1.29
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 56][stp:],
                    "SL Elevation fraction" : chain.samples[:, 57][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-29.pdf')

# Box 1.30
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 58][stp:],
                    "SL Elevation fraction" : chain.samples[:, 59][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-30.pdf')

# Box 1.31
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 60][stp:],
                    "SL Elevation fraction" : chain.samples[:, 61][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-31.pdf')

# Box 1.32
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 62][stp:],
                    "SL Elevation fraction" : chain.samples[:, 63][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-32.pdf')

# Box 1.33
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 64][stp:],
                    "SL Elevation fraction" : chain.samples[:, 65][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-33.pdf')

# Box 1.34
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 66][stp:],
                    "SL Elevation fraction" : chain.samples[:, 67][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-34.pdf')

# Box 1.35
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 68][stp:],
                    "SL Elevation fraction" : chain.samples[:, 69][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation fraction", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-35.pdf')

# Box 1.36
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 70][stp:],
                    "SL Elevation (m)" : chain.samples[:, 71][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
fig.savefig('../Figs/Fig5c/Histogram-2D-36.pdf')

# Box 1.37
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 72][stp:],
                    "SL Elevation (m)" : chain.samples[:, 73][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
plt.savefig("../Figs/Fig5c/Histogram-2D-37.pdf")

# Box 1.38
df = pd.DataFrame({"Age (ka)" : chain.samples[:, 74][stp:],
                    "SL Elevation (m)" : chain.samples[:, 75][stp:]})
fig = sns.jointplot(data=df, x="Age (ka)", y="SL Elevation (m)", kind="hex", palette="colorblind")
plt.savefig("../Figs/Fig5c/Histogram-2D-38.pdf")

# Box 2.1
df = pd.DataFrame({"Erosion rate (mm/yr)" : chain.samples[:, 76][stp:],
                    "Initial slope (%)" : chain.samples[:, 77][stp:]})
fig = sns.jointplot(data=df, x="Erosion rate (mm/yr)", y="Initial slope (%)", kind="hex", palette="colorblind")
fig.savefig("../Figs/Fig5c/Histogram-2D-ER-IS.pdf")

# Box 2.2
df = pd.DataFrame({"Erosion rate (mm/yr)" : chain.samples[:, 76][stp:],
                    "Wave base depth (m)" : chain.samples[:, 78][stp:]})
fig = sns.jointplot(data=df, x="Erosion rate (mm/yr)", y="Wave base depth (m)", kind="hex", palette="colorblind")
fig.savefig("../Figs/Fig5c/Histogram-2D-ER-WB.pdf")

# Box 2.3
df = pd.DataFrame({"Initial slope (%)" : chain.samples[:, 77][stp:],
                    "Wave base depth (m)" : chain.samples[:, 78][stp:]})
fig = sns.jointplot(data=df, x="Initial slope (%)", y="Wave base depth (m)", kind="hex", palette="colorblind")
fig.savefig("../Figs/Fig5c/Histogram-2D-IS-WB.pdf")

# Box 2.4
df = pd.DataFrame({"Erosion rate (mm/yr)" : chain.samples[:, 76][stp:],
                    "Uplift Rate (mm/yr)" : chain.samples[:, 79][stp:]})
fig = sns.jointplot(data=df, x="Erosion rate (mm/yr)", y="Uplift Rate (mm/yr)", kind="hex", palette="colorblind")
fig.savefig("../Figs/Fig5c/Histogram-2D-ER-UR.pdf")

# Box 2.5
df = pd.DataFrame({"Initial slope (%)" : chain.samples[:, 77][stp:],
                    "Uplift Rate (mm/yr)" : chain.samples[:, 79][stp:]})
fig = sns.jointplot(data=df, x="Initial slope (%)", y="Uplift Rate (mm/yr)", kind="hex", palette="colorblind")
fig.savefig("../Figs/Fig5c/Histogram-2D-IS-UR.pdf")

# Box 2.6
df = pd.DataFrame({"Wave base depth (m)" : chain.samples[:, 78][stp:],
                    "Uplift Rate (mm/yr)" : chain.samples[:, 79][stp:]})
fig = sns.jointplot(data=df, x="Wave base depth (m)", y="Uplift Rate (mm/yr)", kind="hex", palette="colorblind")
fig.savefig("../Figs/Fig5c/Histogram-2D-WB-UR.pdf")

plt.show()
