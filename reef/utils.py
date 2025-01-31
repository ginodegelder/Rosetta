# Imports
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from reef import tools as tools
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.abspath('/Users/gino/Documents/SeaLevelHome/Pycharm'))

x, y = tools.readfile("../tests/TopoObs.dat")
x_obs, y_obs = tools.readfile("../tests/TopoObs_Cut.dat")

y = y[y >= 0] # Remove all negative y
y = y[:len(x_obs)] # Same length for y and y_obs
x = x_obs # New x is same as x_obs

print(x)
print(x_obs)
print(y)
print(y_obs)

#xneg=np.zeros((1, 154))
#xneg[:]=x[0:154]

# x= [0, 1, 2, 3, 4, 5]
# y= [0, .1, .2, .3, .4, .5]
# x_obs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# y_obs = [0, .1, .2, .3, .4, .5, .6, .7, .8]
# slopi = 10



#if x[0] <= 0 : neg_x[] = x[]

#y_ex = (slopi/100) * x

# if a < b:
#     x = x_obs
# else:
#     x_obs = x



# def zip_longest(x, y, x_obs, y_obs, slopi):
#
#
#      for i in range(n_params):
#         if len(x) < len(x_obs):
#      else: