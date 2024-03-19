# To replace packaging
# import sys
# sys.path.append("/home/nhedjazi/src/sealevel")

# Change to
import os
import sys
sys.path.insert(0, os.path.abspath('/Users/gino/Documents/SeaLevelHome/Pycharm/mcmc/examples'))



# Imports
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from mcmc.metropolis import Metropolis1dStep

# The (unkown) function to sample from
def fmodel(a,b):
    x1 = (a-2.)/0.5
    y1 = b/2
    x2 = a/1.8
    y2 = (b-1.8)/0.7
    x3 = (a+1)/1.2
    y3 = (b+1)/0.6

    return 0.7*np.exp(-(x1**2+y1**2))+1.3*np.exp(-(x2**2+y2**2))+2.*np.exp(-(x3**2+y3**2));

# Example
x = (4,5)
fmodel(*x)

xx = np.linspace(-4, 4, 100)
yy = np.linspace(-4, 4, 100)
xx, yy = np.meshgrid(xx, yy)
zz = fmodel(xx, yy)
plt.figure(1)
plt.contourf(xx, yy, zz, 20)
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()

# The proposal
def proposal(x, std):
    assert(x.size == std.size)
    xp = np.copy(x)
    # get the size of vector x
    n = x.size
    # Chose a random element in vector x
    i = np.random.randint(0, n)
    # Add a small perturbation
    dx = np.random.normal(0., std[i])
    xp[i] += dx
    return xp

# The log prior
# The prior 0 outside the bounds. Inside the bounds, it is 1/(domain size).
bounds = (-4, 4)

def prior(x):
    logprior = 0
    if any(x[i] < bounds[0] or x[i] > bounds[1] for i in range(x.size)):
        logprior = -np.inf  # log(0)
    else:
        # This is optional because it does not depend on x value, I put it for clarity
        for i in range(x.size):
            logprior += -np.log(bounds[1] - bounds[0])
    return logprior

# The loglikelihood
# In this toy example, we sample fmodel. In future applications, the loglikelihood is the negative of the misfit.

# To compute the value of fmodel at a point x, as seen above, it is simply calling fmodel
def loglike(x):
    a, b = x[0], x[1]
    return np.log(fmodel(a, b))

chain = Metropolis1dStep()
chain.proposal = proposal
chain.prop_S = np.array([2, 2])
chain.logprior = prior
chain.loglikelihood = loglike
chain.show_stats = 10000

# Select a random starting point
a0 = -4 + np.random.rand()*8
b0 = -4 + np.random.rand()*8
x0 = np.array([a0, b0])

# Number of samples
n_samples = 20000
n_tune = 0

# Record some statistics
chain.add_stat("loglikelihood")

# Run the algorithm

chain.run(x0, n_samples, tune=n_tune, tune_interval=1000,
            discard_tuned_samples=False, thin=1)

print("Total duration:")
print(chain.duration)

# Some trace plots
plt.figure(2)
plt.plot(chain.stats["loglikelihood"])
plt.show()

# Plot the posterior samples
plt.figure(3)
plt.scatter(chain.samples[:, 0], chain.samples[:, 1], s=0.1)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.figure(4)
plt.plot(chain.samples[:, 0], chain.samples[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Same plot as a 2D histogram
plt.figure(5)
h = plt.hist2d(chain.samples[:, 0], chain.samples[:, 1], bins=(50,50))
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Marginals are 1D histograms
# 1) for X
plt.figure(6)
sns.histplot(chain.samples[:, 0], kde=True)
plt.xlabel('X')
plt.show()

# 2) for Y
plt.figure(7)
sns.histplot(chain.samples[:, 1], kde=True)
plt.xlabel('Y')
plt.show()