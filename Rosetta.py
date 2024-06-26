# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:49:13 2024

@author: Yannick Boucharat
"""

import os
import sys
sys.path.insert(0, os.path.abspath('./Library'))
sys.path.insert(1, os.path.abspath('./Library/reef'))

# Library imports
from Topo import Plot_FigS4d
from mcmc import misfit as mis
from mcmc.metropolis import Metropolis1dStep
from mcmc import covariance_matrix as cov
from Dicts import Dicos
import tools as tools
from reef_models import reef, reef_platform, reef_eros
from Dict_models import DicoModels
from Inputs import (
    construction, 
    topo_obs, 
    sea_level, 
    construction_params, 
    eros_params, 
    inversion_params
    )

# Python imports
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy import linalg
from datetime import datetime as dtime
import xsimlab as xs
import pprint
from mpi4py import MPI
import tempfile
import shutil

# Get rank and number of processors.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nb_proc = comm.Get_size()

# Inversion parameters.

N_SAMPLES = inversion_params['n_samples']
N_TUNE = inversion_params['n_tune']
TUNE_INTERVAL = inversion_params['tune_interval']
STP = inversion_params['stp']
DX_REEF = inversion_params['dx_reef']
SIGMA = inversion_params['sigma']
CORR_L = inversion_params['corr_l']
IPSTEP = inversion_params['ipstep']
N_CHAINS = inversion_params['n_chains']


# Check if there will be something to plot.
if STP >= N_SAMPLES:
    raise Warning(
        "Starting point for plotting >= number of sample. Nothing will be " \
        "plotted if not restarting from a preexistant model.\n" 
        f"stp = {STP}\n"
        f"n_samples = {N_SAMPLES}"
        )

# Use construction_params if construction = True, eros_param if False
if construction:
    reef_params = construction_params
    # Extract dt REEF and remove it
    DT_REEF = reef_params['dt']
    reef_params.pop('dt')
    # Extracts the first sub dictionnary from reef_params.
    first_sub_dict = next(iter(reef_params.values()))
    # Set "platform" to True or False if 'init__lterr' > 0 or = 0 respectively. 
    if first_sub_dict['init__lterr'][0] > 0:
        platform = True
    else:
        platform = False
        # If no platform, remove platform's params from the dict.
        first_sub_dict.pop('init__lterr', None)
        first_sub_dict.pop('init__zterr', None)
        first_sub_dict.pop('init__sloplat', None)
        first_sub_dict.pop('init__wavelength', None)
        first_sub_dict.pop('init__amplitude', None)
else:
    reef_params = eros_params
    # Extract dt REEF and remove it
    DT_REEF = reef_params['dt']
    reef_params.pop('dt')


# Extracts topographic profiles  

path = topo_obs['topo_path']
# Extracts all the file names in Topo folder
name_files =  [file for file in os.listdir(path) if file.endswith(".dat")]
name_files.sort(key=str.lower)

# Check if there is '.dat' topo files in the folder.
if not name_files:
    raise Exception(
        "No '.dat' files in the Inputs.topo_obs folder."
        )

# Check if number of subdicts in reef_params and topo profiles are equivalent.
if len(reef_params.keys()) != len(name_files):
    raise Exception(
        "Number of subdicts in reef_params must be equal to the " \
        "number of topographic profiles.\n" 
        f"reef_params subdicts = {len(reef_params.keys())}\n"
        f"Number of topographic profiles = {len(name_files)}"
        )

# Check if number of processors used by mpi and number of topo profiles are 
# equivalent.
if nb_proc != len(name_files) * N_CHAINS:
    raise Exception(
        "Number of mpi cores must be equal to the number of topographic " \
        "profiles times number of MCMC chains." 
        f"\nNumber of mpi cores = {comm.Get_size()}"
        f"\nNumber of topographic profiles = {len(name_files)}"
        f"\nNumber of MCMC chains = {N_CHAINS}"
        )

# Creates a dictionary for topographic coordinates. 
# Each key corresponds to one profile.
dict_topo_obs = {}

# num_key corresponds to the "index" of the topographic profiles. It will be 
# used in all the nested dictionaries containing the sets of parameters. This 
# will constrain the separation of the nested dictionary as : first subdict has
# the num_key 0 and contains the reef parameters corresponding to the first 
# topographic profile in the folder "topo_obs['topo_path']", the second subdict 
# has the num_key 1, with reef parameters for second profile etc...
num_key = 0
for filename in name_files: 
    # Extracts x and y in the topo files
    x_obs, y_obs = tools.readfile(topo_obs['topo_path']+'/'+filename)
    # Min and max values
    IPMIN = min(x_obs)  # Min x value from observed topo
    IPMAX = max(x_obs)  # Max x value from observed topo
    Y_OBS_MIN = min(y_obs)  # Min y value from observed topo
    Y_OBS_MAX = max(y_obs)  # Max y value from observed topo
    # Horizontal interpolation, as the misfit will be calculated on y axis.
    # Interpolation of y_obs as a function of x_obs.
    ipobs = interp1d(x_obs, y_obs)  
    # Create an array from x min to x max with the step defined in "Inputs".
    x_obs_n = np.arange(IPMIN, IPMAX, IPSTEP) 
    # Interpolate y_obs to fit x_obs_n dimension.
    y_obs_n = ipobs(x_obs_n) 
    # Puts values in the dictionnary 
    dict_topo_obs[f"topo_obs_{num_key}"] = {
        f"x_obs_{num_key}" : x_obs_n,
        f"y_obs_{num_key}" : y_obs_n,
        f"ipmin_{num_key}" : IPMIN,
        f"ipmax_{num_key}" : IPMAX,
        f"y_obs_min_{num_key}" : Y_OBS_MIN,
        f"y_obs_max_{num_key}" : Y_OBS_MAX
        }
    num_key += 1


# Matrixes

# Number of matrix elements, round up to full number.
#N1 = int(-(-(IPMAX - IPMIN) // IPSTEP)) 
DX_COV = DX_REEF  # Here equal to 1, dependent of dx used in the reef model .
GAMMA = 1  # Exponent in the kernel. 1 for laplacian, 2 for gaussian.

# Creates a dictionary of matrixes.
# Each key corresponds to one profile.
dict_matrix = {}

for key in dict_topo_obs.keys():
    # Extracts the number of the subdict 
    num_key = int(''.join(filter(str.isdigit, key)))
    # Computing inverse covariance.
    # Number of matrix elements, round up to full number.
    N1 = int(-(-(dict_topo_obs[key][f'ipmax_{num_key}'] - 
                 dict_topo_obs[key][f'ipmin_{num_key}']) // IPSTEP)) 
    # First matrix: the physical model error. Computes the covariance.
    covar = cov.exponential_covar_1d(N1, SIGMA, CORR_L, dx = DX_COV, 
                                     gamma = GAMMA, truncate = None) 
    # Second one, measurement error. Computes the inverse covariance.
    icovar = linalg.inv(covar) 
    # Puts matrixes in "dict_matrix"
    dict_matrix[f"matrix_profile_{num_key}"] = {
        f"covar_{num_key}" : covar,
        f"icovar_{num_key}" : icovar
        }


# Extracts SL nodes and reef parameters.  
# Converts them in panda dataframes and separates fixed and free nodes or 
# parameters. Every variables in "Inputs.py" without values in the first column
# will be considered as fixed, the others will be free. 

# For SL. 
# One SL history : no need of dictionary.

# Extracts the time length of the simulations.
TSTART = max(sea_level.keys()) 
# Creates the SL dataframe
df_SL = pd.DataFrame.from_dict(sea_level, orient = "index", 
                               columns = ['t_min', 't_max', 'step_t', 
                                          'e_start', 'e_min', 'e_max', 
                                          'step_e'])

# Puts t_start in first column, to have normal indexes.
df_SL.insert(0, 't_start', df_SL.index)
df_SL.reset_index(drop = True, inplace = True)

# Creates two empty dataframes for SL. 
# First one for fixed nodes, second one for free nodes.
df_SL_fixed = pd.DataFrame(columns = ['e_start'])  
df_SL_free = pd.DataFrame(columns = df_SL.columns)  

for i in df_SL.index:
    # Extracts the fixed SL nodes and puts them in "df_SL_fixed" dataframe
    if df_SL.loc[i].isnull().values.any() == True:
        df_SL_fixed = pd.concat([df_SL_fixed, df_SL.loc[[i], ['e_start']]]) 
    # Extracts the free SL nodes and puts them in "df_SL_free" dataframe
    else:
        df_SL_free.loc[i] = df_SL.loc[i]


# For reef parameters. 
# One set of free parameters for each topographic profile : need dictionaries.

# Dataframe for fixed reef parameters.
df_reef_fixed = pd.DataFrame()
# Dictionary of dataframes with fixed and free parameters for each profile.
dict_df_reef = {}
# Dictionary of dataframes with the free parameters for each profile.
dict_df_reef_free = {}

# Fill the reef dictionaries and the dataframe "df_reef_fixed".
num_key = 0
for key, subdict in reef_params.items():
    # Convert hwb in float for forward model
    subdict['eros__hwb'][0] = float(subdict['eros__hwb'][0])
    # Generating the dataframes from the subdicts.
    # The first subdict (num_key = 0) contains both fixed and free parameters.
    if num_key == 0: 
        # Create a dataframe with all parameters from the first subdict.
        df_reef_0 = pd.DataFrame.from_dict(
            reef_params[key], orient = 'index', 
            columns = ['Starting_Point', 'Min', 'Max', 'Step Size']
            )
        # Puts "df_reef_0" in "dict_df_reef"
        dict_df_reef["df_reef_0"] = df_reef_0
        # Create empty dataframe with same columns as df_reef_0. Will contain
        # only free parameters.
        df_reef_free_0 = pd.DataFrame(columns = df_reef_0.columns)

        # Separate fixed and free parameters from df_reef_0
        for subkey in subdict.keys():            
            # Extracts the fixed parameters.
            if df_reef_0.loc[subkey].isnull().values.any() == True:
                # Put fixed parameters in "df_reef_fixed".
                df_reef_fixed = pd.concat(
                    [df_reef_fixed, 
                     df_reef_0.loc[[subkey], ['Starting_Point']]]
                    )
            
            #Extracts the free parameters.
            else:
                # Puts free parameters in "df_reef_free".
                df_reef_free_0.loc[subkey] = df_reef_0.loc[subkey]
        
        # Put "df_reef_free_0" in the dictionary of dataframes.
        dict_df_reef_free["df_reef_free_0"] = df_reef_free_0

        
    # The next subdicts (i > 0) will only contain free parameters.
    else:
        # Create the name of the dataframe. 
        df_name = f"df_reef_free_{num_key}"
        # Create the dataframe with the generated name.
        df_name = pd.DataFrame.from_dict(reef_params[key], orient = 'index', 
                                          columns = ['Starting_Point', 'Min', 
                                                     'Max', 'Step Size'])
        # Sort by indexes to have all dataframes in the same format.
        df_name = df_name.reindex(df_reef_free_0.index)
        # Put the dataframe in "dict_df_reef_free".
        dict_df_reef_free[f"df_reef_free_{num_key}"] = df_name
        # Combine fixed and free parameters in dataframes put in "dict_df_reef"
        for param in subdict:
            # Update df_reef_0 (used as a base)
            df_reef_0 = df_reef_0.replace(
                df_reef_0.loc[param], df_name.loc[param]
                )
            # Puts the updated dataframe in dict_df_reef
            dict_df_reef[f"df_reef_{num_key}"] = df_reef_0

        
    # At the end of the loop, dict_df_reef_free has the same format as 
    # "reef_params" dictionary but the subdicts are replaced by dataframes.
    # The dictionary "dict_df_reef" is the same as "dict_df_reef_free" but 
    # each dataframes contains the fixed and free parameters corresponding 
    # to each profile.
        
    num_key += 1

# Check if numbers of free reef parameters are consistent between profiles.
for key in dict_df_reef_free.keys():
    if len(dict_df_reef_free['df_reef_free_0']) != len(dict_df_reef_free[key]):
        raise Exception( 
            "Not same number of free reef parameters for each topographic " \
            "profile in Inputs.reef_params dictionnary."
            )
    else:
        continue
    
    
# Proposal, starting points and bounds lists.

# For SL.
# Proposal lists for time and elevations.
prop_t_SL = df_SL_free['step_t'].values.tolist()
prop_e_SL = df_SL_free['step_e'].values.tolist()
# Starting points lists for time and elevations
SL_t_x0 = df_SL_free['t_start'].values.tolist()
SL_e_x0 = df_SL_free['e_start'].values.tolist()
# Bounds lists for time and elevations.
bounds_t_SL = np.array([df_SL_free['t_min'].values.tolist(),
                        df_SL_free['t_max'].values.tolist()]).T
bounds_e_SL = np.array([df_SL_free['e_min'].values.tolist(),
                        df_SL_free['e_max'].values.tolist()]).T

# Concatenate in dictionaries SL proposal, starting points and bounds lists  
# with reef parameters proposal, starting points and bounds values.

# Create a dict for all starting points.
dict_start = {}
# Create a dict for all bounds.
dict_bounds = {}
# Create a dict for all proposals.
dict_prop = {}

for key in dict_df_reef_free.keys(): 
    # Fill dictionnaries with free reef parameters and free SL nodes values of 
    # starting_point, bounds and step_size.
    # Output :
        # dict = { 'profile_i': array([ first_t_SL, first_e_SL, ..., 
        #   last_t_SL, last_e_SL, first_reef_param, ..., last_reef_param ]),
        #   'profile_i+1': array([ first_reef_param, ..., last_reef_param ]) }
        # With i number of profiles to inverse, so i number of keys in "dict".
        # Reef params values in the array correspond to free reef params 
        # involved for the topo profile_i.
    
    # Extracts the number of the subdict key
    num_key = int(''.join(filter(str.isdigit, key)))
    # For i = 0, extract SL and free reef params values
    if num_key == 0:
        # Fill "dict_start"
        dict_start[f"start_{num_key}"] = np.hstack([
            np.ravel([SL_t_x0, SL_e_x0], 'F'), 
            dict_df_reef_free[key]['Starting_Point'].values.tolist()
            ])
        
        # Fill "dict_bounds"
        dict_bounds[f"bounds_{num_key}"] = np.vstack((
            np.concatenate([*zip(bounds_t_SL, bounds_e_SL)]),
            np.array(
                [dict_df_reef_free[key]['Min'].values.tolist(),
                 dict_df_reef_free[key]['Max'].values.tolist()]
                ).T
            )) 
        
        # Fill "dict_prop"
        dict_prop[f"prop_{num_key}"] = np.hstack(
            [np.ravel([prop_t_SL,prop_e_SL], 'F'), 
             dict_df_reef_free[key]['Step Size'].values.tolist()])
        
    # For i > 1, only values for free reef parameters are added 
    else:
        dict_start[f"start_{num_key}"] = np.hstack(
            [dict_df_reef_free[key]['Starting_Point'].values.tolist()]
            )
        
        dict_bounds[f"bounds_{num_key}"] = np.vstack((
            np.array(
                [dict_df_reef_free[key]['Min'].values.tolist(),
                 dict_df_reef_free[key]['Max'].values.tolist()]
                ).T
            )) 
        
        dict_prop[f"prop_{num_key}"] = np.hstack(
            dict_df_reef_free[key]['Step Size'].values.tolist())
            
        
# Concatenate everything in lists. The inversion code takes as inputs only 1D
# lists and arrays. Previous dictionaries are used for clarity and flexibility.

x0 = []
for key in dict_start:
    x0.append(dict_start[key])
x0 = np.concatenate(x0)

bounds = []
for key in dict_bounds:
    bounds.append(dict_bounds[key])
bounds = np.concatenate(bounds)

prop_S = []
for key in dict_prop:
    prop_S.append(dict_prop[key])
prop_S = np.concatenate(prop_S)

# Initialize other chains if N_CHAINS > 1
# Create a list of sub_master cores
sub_master = []
# Create a 2D list of chains
chains = [x0]
for i in range(1, N_CHAINS):
    # One sub_master core for each chain, eg. 3 profiles and 3 chains : 9 cores
    # with sub_master = [3,6]
    sub_master.append(i * len(name_files))
    sub_chain = []
    # Add a random perturbation to the parameters based on its standard dev.
    for j in range(len(x0)):
        perturb = np.random.uniform(-1, 1) * prop_S[j]
        param_j = x0[j] + perturb
        sub_chain.append(param_j)
    # Add the sub_chain to chains
    chains.append(sub_chain)

# Creates a temporary directory to store SL files. Will be erased at the end.
if rank == 0:
    # Creates the directory and store its path.
    temp_SL_dir = tempfile.mkdtemp(prefix='Tempdir_SL_', 
                                   dir='./Library/RSLcurves')
else:
    temp_SL_dir = None

# Send the dir path to other ranks.
temp_SL_dir = comm.bcast(temp_SL_dir, root=0)
# Extracts only the name of the temporary directory.
temp_SL_dir_name = os.path.basename(temp_SL_dir)


# Functions

def proposal(x, std):
    """
    Computes the proposal : add a random perturbation according to std on one 
    value in x.

    Parameters
    ----------
    x : 1D list
        Free reef parameters and free SL nodes.
    std : 1D list 
        Standard deviations (prop_S) associated to x.

    Returns
    -------
    xp : 1D list
        New list of parameters with a pertubed value.

    """
    assert (x.size == std.size)
    xp = np.copy(x)
    # Get the size of vector x
    N = x.size
    # Chose a random element in vector x
    i = np.random.randint(0, N)
    # Add a small perturbation
    dx = np.random.normal(0., std[i])
    xp[i] += dx
    
    return xp


def prior(x):
    """
    Computes the log prior probability function.
    
    The prior is 0 outside the bounds. Inside the bounds, 
    it is 1/(domain size).

    Parameters
    ----------
    x : 1D list 
        Free reef parameters and free SL nodes.

    Returns
    -------
    logprior : float
        Log prior probability value.

    """
    logprior = 0
    n_params = len(x)
    
    for i in range(n_params):
        # If the new value is out of the bounds, the prior, and thus the 
        # posterior probability are null
        if x[i] < bounds[i, 0] or x[i] > bounds[i, 1]:
            logprior = - np.inf  # log(0)
        else:
            # This is optional because it does not depend on x value, 
            # I put it for clarity
            for i in range(x.size):
                logprior += - np.log(bounds[i, 1] - bounds[i, 0])
                
    return logprior


def param(x, dict_save_vars):
    """
    Change the value of the free parameters in the dataframes and returns a 
    nested dictionary with each subdict corresponding to one topographic 
    profile.
    
    Extracts the values in x and change the corresponding values in the 
    dataframes. Converts the new SL dataframe in a .dat file and the new reef
    dataframes in dictionaries. The reef dictionaries are stored as subdicts
    in the nested dictionary "dict_input_vars".

    Parameters
    ----------
    x : 1D list 
        Free reef parameters and free SL nodes.
    dict_save_vars : nested dictionnary
        Each sub-dictionnary is a save of the reef parameters involved in the 
        last iteration. One sub-dictionnary for on core/topographic profile.

    Returns
    -------
    dict_input_vars : nested dictionnary 
        Each subdict contains all the parameters for one reef simulation. 
        The number of subdicts corresponds to the number of topographic 
        profiles to inverse.
    dict_save_vars : nested dictionnary
        Updated save of the parameters. Same format as the input.
    tnew : 1D numpy array 
        New time values in SL history.
    enew : 1D numpy array 
        New SL elevations corresponding to tnew.
    focus_run : int or str
        Asks the model to run only one core or all cores.
        If only one reef parameter has been changed, returns an int, 
        representing the rank of the core to run. If a sea-level parameter has
        been changed, return a str ('ALL'), asking the model to run all cores
        for this iteration.

    """    
    # Change values in the dataframes.
    # For reef parameters.   
    j = 0
    for key, subdict in dict_df_reef_free.items():
        # Extracts the number in the key.
        num_key = int(''.join(filter(str.isdigit, key)))
        # Extracts the names of the free reef parameters.
        for param in df_reef_free_0.index:
            # Replace the values in dict_df_reef by the values in x.
            dict_df_reef[f"df_reef_{num_key}"].loc[param, 'Starting_Point'] = (
                x[len(df_SL_free) * 2 + j]
                )
            j += 1
    
    # Create the dictionary dict_input_vars to store the changed reef 
    # dataframe. 
    dict_input_vars = {}
    
    i = 0
    for key in dict_df_reef.keys():
        dict_input_vars[f"input_vars_{i}"] = (
            dict_df_reef[f"df_reef_{i}"]['Starting_Point'].to_dict()
            )
        i += 1
    
    # Compare the new params with the last one. If one reef param changed, only
    # associated core will run, else, SL has changed, every cores need to run.
    
    # Make a copy of dict_input_vars and remove 'SL', as the path of the 
    # tempdir is always different.
    dict_input_check = dict_input_vars.copy()
    for key in dict_input_check.keys():
        dict_input_check[key].pop('SLstory__RSLin', None)
        
        # No need normally, but unfixed bug.
        try:
            dict_save_vars[key].pop('SLstory__RSLin', None)
        # At first iteration, dict_save_vars is empty.
        except KeyError:
            continue
    
    # At first iteration, saves are empty, all cores run.
    if not dict_save_vars:
        focus_run = 'ALL'
    
    # If both dicts are equal, SL has been changed, all cores run.
    elif dict_save_vars == dict_input_check:
        focus_run = 'ALL'
        
    # If different, extracts the subdict that has been changed.
    else:
        # Extraction of the changed subdict's name
        diff_subdicts = [key for key in dict_input_check.keys() \
                         if dict_input_check[key] != dict_save_vars[key]]
        # Keeps only the subdict's number, which is equal to the rank of the 
        # core to run.
        focus_run = int(''.join(filter(str.isdigit, diff_subdicts[0])))
        
    # Update the saving dictionnary. Unknown bug : adds 'SLstory__RSLin' into
    # "dict_save_vars". Ignores pop() used to on dict_input_check.
    dict_save_vars = dict_input_check.copy()
    
    # For SL history.
    j = 0
    for i in df_SL_free.index :  # Extract only the free SL parameters.
        # Change the values of the free SL parameters in the SL dataframe.
        df_SL['t_start'].values[i] = x[j * 2]
        df_SL['e_start'].values[i] = x[(j * 2) + 1]
        j += 1
    
    # Create two lists for SL time and elevation with the changed SL dataframe.
    t = df_SL['t_start'].values.tolist()
    e = df_SL['e_start'].values.tolist()
        
    # Creating the filename of the new SL file. Rank dependant to avoid 
    # overwriting.
    SL_name = "SL_new"+str(rank)+".dat"
    # Adds the path of the temporary directory.
    SL_Path = temp_SL_dir + '/' + SL_name
    
    # Put the name of the new SL file in input_vars.    
    for key in dict_input_vars.keys():
        dict_input_vars[key]['SLstory__RSLin'] = (
            temp_SL_dir_name + '/' + SL_name
            )
        
    # Interpolation of e with respect to t.
    pc = interpolate.PchipInterpolator(t, e, axis=0, extrapolate=None)
    # New time range.
    tnew = np.arange(TSTART - 1, - 1, - 1) 
    # Interpolate SL elevations according to tnew range.
    enew = pc(tnew)  
    # Put them in a 2D array.
    SL_new = np.array([tnew, enew])  
    # Stack SL values in two columns.
    SL_story = np.column_stack((SL_new)) 
    # Save SL history in a .dat file. SAVE IT IN A TEMPORARY FILE
    np.savetxt(SL_Path, SL_story, delimiter = '\t') 
    
    return dict_input_vars, dict_save_vars, tnew, enew, focus_run


def misfit(y_n, y_obs_n, icovar_i):
    """
    Computes the misfit on y axis between the constructed (y_n) and the 
    observed (y_obs_n) topography.

    Parameters
    ----------
    y_n : 1D array
        Modeled elevations.
    y_obs_n : 1D array
        Observed elevations.
    icovar_i : 2D numpy array
        Inverse covariance.

    Returns
    -------
    fit : float
        Squared Mahalanobis distance misfit.

    """
    fit = mis.sqmahalanobis(y_n, y_obs_n, icovar_i)
    
    return fit


def align(x, y, ipmin_i, ipmax_i, y_obs_min):
    """
    Cut the simulated array to fit with topo_obs dimensions and avoid offsets.
    
    Based on the y axis : extracts the min y value in the observed topo, and 
    when this value is reached on the y axis of the constructed topo, this will
    be the beginning of the profile. Extract after the x length of the observed
    topo and adjust the constructed topo to this length. 
    To resume, if min y value observed is -25 m, and its horizontal length is 
    5 km, the constructed topo will be cut at the first -25 m on y axis, and it 
    will be cut 5 km after on x axis.

    Parameters
    ----------
    x : 1D array
        Horizontal values of the simulation.
    y : 1D array
        Vertical values of the simulation.
    IPMIN : float
        Minimum x value from the observed topography.
    IPMAX : float
        Maximum x value from the observed topography.
    y_obs_min : float
        Minimum y value from the observed topography.

    Returns
    -------
    x_n : 1D array
        New horizontal values for the plot.
    y_n : 1D array
        New vertical values for the plot.
    

    """
    # Extract the index in y of the first element >= to the min value in y_obs.
    index_min = np.argmax(y >= y_obs_min) 
    # Use this index for the x starting value.
    x_start = x[index_min]
    
    ipmod = interp1d(x, y) # Interpolate y as a function of x.
    # Generate new continuous x values from x_start to max x value in topo_obs.
    # with the step defined in Inputs.inversion_params.
    x_n = np.arange(x_start, x_start - ipmin_i + ipmax_i, IPSTEP)
    y_n = ipmod(x_n) # Interpolation of the vertical values on new x_n axis.
    x_n = x_n - x_n[0] # Remove the offset to have x axis starting at 0.
    
    return x_n, y_n


def run_reef(input_vars):
    """
    Runs the reef construction model.
    
    Uses one subdict "input_vars_n" from the nested dictionnary 
    "dict_input_vars" created by param(x) function.

    Parameters
    ----------
    input_vars : dictionnary
        Full set of parameters for one reef simulation.

    Returns
    -------
    x : 1D numpy array
        Horizontal values of the constructed profile
    y : 1D numpy array
        Elevations of the constructed profile 

    """
    # Starting time of the model
    tmax = TSTART * 1e3 
    # print(tmax)
    
    # Set the model type. 
    # If coral construction and platform = True, model_type = reef_platform.
    # If coral construction = True and platform = False, model_type = reef.
    # If coral construction and platform = False, model_type = reef_eros.
    if construction:
        if platform:
            model_type = reef_platform
            model_name = 'reef_platform'
        else:
            model_type = reef
            model_name = 'reef'
    else:
        model_type = reef_eros
        model_name = 'reef_eros'
    
    # Create the setup for the simulation
    ds_in = xs.create_setup(
        model = model_type,
        clocks = {
            'time' : np.arange(0., tmax + DT_REEF, DT_REEF)
    },
        master_clock = 'time',
        # Uses the subdict created by param(x) function.
        input_vars = input_vars,  
        # In this configuration of output_vars, only the last topo profile 
        # is saved
        output_vars = {
            'init__x'       : None,
            'profile__z'    : None,
            'sealevel__asl' : None,
            'profile__xmin' : None,
            'profile__xmax' : None,
        }
    )
    
    # Set attributes.
    ds_in.attrs['model_name'] = model_name
    
    # Run    
    #t0 = dtime.now()
    
    with DicoModels().models[ds_in.model_name]:
        ds = (ds_in.xsimlab.run())
        
    #print("simu rank ", rank, " duration : ", dtime.now() - t0)
    
    # Extracts the last topo profile.
    x = ds.x[:].values
    y = ds.profile__z[:].values
    
    return x, y


def loglike(x, dict_save_run, dict_save_vars):
    """
    Ask run_platform, misfit and align functions, computes the loglike 
    and make predictions (posterior predictive).
    
    In future applications, the loglikelihood is the negative of the misfit.

    Parameters
    ----------
    x : 1D list
        Reef parameters (x0)
    dict_save_run : nested dictionnary
        Each sub-dictionnary is a save of the outputs values from the 
        simulation of the last iteration. One sub-dictionnary for one 
        core/topographic profile.
    dict_save_vars : nested dictionnary
        Each sub-dictionnary is a save of the reef parameters involved in the 
        last iteration. One sub-dictionnary for one core/topographic profile.

    Returns
    -------
    -0.5 * fit : float
        Loglike value
    predictions : dictionnary
        Predictions for the posterior
    dict_save_run : nested dictionnary
        Updated save of the simulations outputs. Same format as the input.
    dict_save_vars : nested dictionnary
        Unchanged from the input. Here just to keep it the loop and reuse for 
        next iteration.

    """
    # Extracts the parameters for reef model and update the saves and focus_run 
    # argument.
    dict_input_vars, dict_save_vars, tnew, enew, focus_run = \
        param(x, dict_save_vars)
    
    sum_fit = 0 
    predictions = {}
    
    if focus_run == 'ALL':
        for key in dict_input_vars.keys():
            num_key = int(''.join(filter(str.isdigit, key)))
            
            # Selects the core.
            if rank == num_key:  
                # Run the reef simulation on the selected core.
# =============================================================================
#                 try:
#                     x, y = run_reef(dict_input_vars[key])
#                 except Exception:
#                     report_REEF_error()
#                     return None, None, dict_save_run, dict_save_vars
# =============================================================================
                x, y = run_reef(dict_input_vars[key])
                # Empty dict_save_run before new saves.
                dict_save_run = {}
                # Update dict_save_run.
                dict_save_run = {f"rank_{rank}" : {
                    "x" : x,
                    "y" : y
                        }}
                # Extracts min and max x values from observed topography.
                ipmin_i = dict_topo_obs[f"topo_obs_{num_key}"] \
                    [f"ipmin_{num_key}"]
                ipmax_i = dict_topo_obs[f"topo_obs_{num_key}"] \
                    [f"ipmax_{num_key}"]
                # Minimum y_obs value.
                y_obs_min_i = dict_topo_obs[f"topo_obs_{num_key}"] \
                    [f"y_obs_min_{num_key}"]
                # Align the simulation with topo_obs.
                x_n, y_n = align(x, y, ipmin_i, ipmax_i, y_obs_min_i)
                # Store them in a dictionary for posterior predictions.
                dict_pred = {
                    f"x_{num_key}" : x_n,
                    f"y_{num_key}" : y_n
                    }
                # Extracts the inverse covariance matrix.
                icovar_i = dict_matrix[f"matrix_profile_{num_key}"] \
                   [f"icovar_{num_key}"]
                # Extracts the observed elevation.
                y_obs_n = dict_topo_obs[f"topo_obs_{num_key}"] \
                    [f"y_obs_{num_key}"]
                # Computes the misfit.
                fit = misfit(y_n, y_obs_n, icovar_i)
        
    elif type(focus_run) == int:
        for key in dict_input_vars.keys():
            num_key = int(''.join(filter(str.isdigit, key)))
            
            if rank == focus_run and rank == num_key:
                # Run the reef simulation on the selected core.
                x, y = run_reef(dict_input_vars[key])
                # Empty dict_save_run before new saves.
                dict_save_run = {}
                # Update dict_save_run.
                dict_save_run = {f"rank_{rank}" : {
                    "x" : x,
                    "y" : y
                        }}
                # Extracts min and max x values from observed topography.
                ipmin_i = dict_topo_obs[f"topo_obs_{num_key}"] \
                    [f"ipmin_{num_key}"]
                ipmax_i = dict_topo_obs[f"topo_obs_{num_key}"] \
                    [f"ipmax_{num_key}"]
                # Minimum y_obs value.
                y_obs_min_i = dict_topo_obs[f"topo_obs_{num_key}"] \
                    [f"y_obs_min_{num_key}"]
                # Align the simulation with topo_obs.
                x_n, y_n = align(x, y, ipmin_i, ipmax_i, y_obs_min_i)
                # Store them in a dictionary for posterior predictions.
                dict_pred = {
                    f"x_{num_key}" : x_n,
                    f"y_{num_key}" : y_n
                    }
                # Extracts the inverse covariance matrix.
                icovar_i = dict_matrix[f"matrix_profile_{num_key}"] \
                   [f"icovar_{num_key}"]
                # Extracts the observed elevation.
                y_obs_n = dict_topo_obs[f"topo_obs_{num_key}"] \
                    [f"y_obs_{num_key}"]
                # Computes the misfit.
                fit = misfit(y_n, y_obs_n, icovar_i)
            
            else:                
                # Selects the core.
                if rank == num_key:
                    # Uses the saved outputs from last run.
                    x = dict_save_run[f"rank_{rank}"]["x"]
                    y = dict_save_run[f"rank_{rank}"]["y"]
                    # Empty dict_save_run before new saves.
                    dict_save_run = {}
                    # Update dict_save_run. No need here but add it for 
                    # gathering simplicity.
                    dict_save_run = {f"rank_{rank}" : {
                        "x" : x,
                        "y" : y
                            }}
                    # Extracts min and max x values from observed topography.
                    ipmin_i = dict_topo_obs[f"topo_obs_{num_key}"] \
                        [f"ipmin_{num_key}"]
                    ipmax_i = dict_topo_obs[f"topo_obs_{num_key}"] \
                        [f"ipmax_{num_key}"]
                    # Minimum y_obs value.
                    y_obs_min_i = dict_topo_obs[f"topo_obs_{num_key}"] \
                        [f"y_obs_min_{num_key}"]
                    # Align the simulation with topo_obs.
                    x_n, y_n = align(x, y, ipmin_i, ipmax_i, y_obs_min_i)
                    # Store them in a dictionary for posterior predictions.
                    dict_pred = {
                        f"x_{num_key}" : x_n,
                        f"y_{num_key}" : y_n
                        }
                    # Extracts the inverse covariance matrix.
                    icovar_i = dict_matrix[f"matrix_profile_{num_key}"] \
                       [f"icovar_{num_key}"]
                    # Extracts the observed elevation.
                    y_obs_n = dict_topo_obs[f"topo_obs_{num_key}"] \
                        [f"y_obs_{num_key}"]
                    # Computes the misfit.
                    fit = misfit(y_n, y_obs_n, icovar_i)                    
    
    # Collects all run saves. Allgather argument concatenates values in a list.
    list_dict_save_run = comm.allgather(dict_save_run)
    # Remove the list.
    dict_save_run = {key: value for dicos in list_dict_save_run 
                     for key, value in dicos.items()}

    # Collects the posterior predictions dictionnaries on each core.
    list_dict_pred = comm.allgather(dict_pred)
    # Concatenate all these dictionnaries in the "predictions" dictionary.
    for dict in list_dict_pred:
        for key in dict.keys():
            predictions[key] = dict[key]
    # Store SL posterior predictions
    predictions["t"] = tnew
    predictions["e"] = enew
    
    # Collects the misfits calculated on each core.
    list_fit = comm.allgather(fit)
    # Computes the sum of the misfits .
    sum_fit = sum(list_fit)
    
    return -0.5 * sum_fit, predictions, dict_save_run, dict_save_vars

# Run 

# Initialisation : update the different internal functions of the inversion 
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

# Create the Out folder
if rank == 0:    
    # Extracts time of the form day-month-year_hour.min.
    time_output = dtime.now().strftime('%d-%m-%Y_%H.%M')     
    # Creates a unique name with N_SAMPLES, SIGMA IPSTEP and time_output.
    Folder_name = ('Figs_' + str(N_SAMPLES) + '_sig.' + str(SIGMA) + '_ip.' + 
                 str(IPSTEP) + '_' + time_output)
    # Creates a path with Folder_name.
    Folder_path = os.path.join(os.getcwd(), 'Outs/FigS4d/' + Folder_name)
    # Creates the directory for the Outs.
    os.makedirs(Folder_path)

    # Creates a folder to store the outputs raw data.
    Df_folder_path = os.path.join(os.getcwd(), Folder_path + '/Dataframes')
    os.makedirs(Df_folder_path)
    
    for other_rank in range(1, nb_proc):
        comm.send(Df_folder_path, dest=other_rank)

else:
    Df_folder_path = comm.recv(source = 0)

# Run the algorithm
chain.run(chains, N_SAMPLES, Df_folder_path, tune = N_TUNE, 
          tune_interval = TUNE_INTERVAL,
          discard_tuned_samples = False, thin = 1)


# =============================================================================
# Plotting
# =============================================================================

if rank == 0:
    
    # Erase the temporary SL folder.
    shutil.rmtree(temp_SL_dir)
    
    # Print duration of the algorithm
    print("\ntotal duration:")
    print(chain.duration)
    
    # Some statistics, records of best SL, profile... 
# =============================================================================
#     # Extracts time of the form day-month-year_hour.min.
#     time_output = dtime.now().strftime('%d-%m-%Y_%H.%M')     
#     # Creates a unique name with N_SAMPLES, SIGMA IPSTEP and time_output.
#     Folder_name = ('Figs_' + str(N_SAMPLES) + '_sig.' + str(SIGMA) + '_ip.' + 
#                  str(IPSTEP) + '_' + time_output)
#     # Creates a path with Folder_name.
#     Folder_path = os.path.join(os.getcwd(), 'Outs/FigS4d/' + Folder_name)
#     # Creates the directory for the Outs.
#     os.makedirs(Folder_path)
# 
#     # Creates a folder to store the outputs raw data.
#     Df_folder_path = os.path.join(os.getcwd(), Folder_path + '/Dataframes')
#     os.makedirs(Df_folder_path)
# =============================================================================
    
    # Save Inputs file in Figs folder.
    with open(Folder_path + '/AA-Inputs.txt', 'w') as f:
        f.write("Construction = ")
        pprint.pprint(construction, stream = f)
        f.write("\n")
        
        f.write("sea_level = \n")
        pprint.pprint(sea_level, stream = f)
        f.write("\n")
    
        f.write("reef_params = \n")
        pprint.pprint(reef_params, stream = f)
        f.write("\n")
    
        f.write("inversion_params = \n")
        pprint.pprint(inversion_params, stream = f)
        f.write("\n")
    
        f.write("topo_obs = \n")
        pprint.pprint(topo_obs, stream = f)
        f.write("\n")
    
    # Some trace plots for statistics.
    # Creates a folder to store stats figures.
    Stats_folder_path = os.path.join(os.getcwd(), Folder_path + '/Stats')
    os.makedirs(Stats_folder_path)
    
    # Evolution of the loglike at each step.
    fig = plt.figure()
    plt.plot(chain.stats["loglikelihood"][1:])
    fig.savefig(Stats_folder_path + '/Stats-Loglikelihood.png')
    
    # Evolution of the standard deviation at each steps (constant if N_TUNE = 0 
    # or TUNE_INTERVAL >= N_SAMPLES).
    fig = plt.figure()
    plt.plot(chain.stats["prop_S"][1:])
    fig.savefig(Stats_folder_path + '/Stats-prop_S.png')
    
    # Acceptance ratio of the simulations.
    fig = plt.figure()
    plt.plot(chain.stats["accept_ratio"][1:])
    fig.savefig(Stats_folder_path + '/Stats-accept_ratio.png')
    
    # Acceptance ratio of the parameters.
    fig = plt.figure()
    plt.plot(chain.stats["parameter_accept_ratio"][1:])
    fig.savefig(Stats_folder_path + '/Stats-parameter_accept_ratio.png')
    
    plt.close('all')
    
    # Profile plot.
    best = np.argmax(chain.stats["loglikelihood"][STP:])
    
    # Remove extensions from topo filenames 
    name_files_short = [os.path.splitext(filename)[0] 
                        for filename in name_files]
    for i in range(len(name_files_short)):
        Sub_folder_name = name_files_short[i]
        Sub_folder_path = os.path.join(os.getcwd(), Folder_path + 
                                       '/' + Sub_folder_name)
        os.makedirs(Sub_folder_path)
        x_n = chain.posterior_predictive[f"x_{i}"][0, 0, :]
        y_n = chain.posterior_predictive[f"y_{i}"][0, STP:, :]
        x_obs = dict_topo_obs[f"topo_obs_{i}"][f"x_obs_{i}"]
        y_obs = dict_topo_obs[f"topo_obs_{i}"][f"y_obs_{i}"]
        fig, fig2 = Plot_FigS4d.profile_y(
            x_n, y_n, x_obs, y_obs, best, i, Sub_folder_path
            )
        plt.close()
    
    all_loglikes = chain.stats["loglikelihood"][STP:, :]
    best_loglike = all_loglikes[best, :]
    np.savetxt(Folder_path + '/BestLogLike.txt', best_loglike)
    
    # Sea-level plot 
    SL_folder_path = os.path.join(os.getcwd(), Folder_path + '/SL')
    os.makedirs(SL_folder_path)
    xsl = np.arange(0, TSTART, 1)
    #xsl = chain.posterior_predictive["t"][0, :, :][STP:]
    ysl = chain.posterior_predictive["e"][0, :, :][STP:]
    fig, fig2 = Plot_FigS4d.sealevel(xsl, ysl, best, SL_folder_path)
    
    mean = np.mean(ysl, axis = 0)
    median = np.percentile(ysl[:, :], 50, axis = 0)
    best_sl = ysl[best, :]
    np.savetxt(SL_folder_path + '/MeanSL.txt', mean)
    np.savetxt(SL_folder_path + '/MedianSL.txt', median)
    np.savetxt(SL_folder_path + '/BestSL.txt', best_sl)
    
    plt.close('all')
    
    # Plotting histograms with all free parameters
    
    # Plotting the free SL nodes
    j = 0
    for i in df_SL_free.index :
        # Extracts t_start for the name of the plot.
        istr = str(df_SL_free.loc[i]['t_start'])
        # Creates a dataframe with the output values for SL from STP point.
        df = pd.DataFrame({
            "Age (ka)" : chain.samples[:, j*2][STP:], 
            "SL Elevation (m)" : chain.samples[:, j*2+1][STP:]
            })
        # Dataframe to save, with all values.
        df_save = pd.DataFrame({
            "Age (ka)" : chain.samples[:, j*2][:], 
            "SL Elevation (m)" : chain.samples[:, j*2+1][:]
            })
        # Store the dataframe.
        df_save.to_pickle(Df_folder_path + '/df_SL_' + istr + 'ky.pkl')
        # Plot.
        fig = sns.jointplot(data = df, x = "Age (ka)", y = "SL Elevation (m)", 
                            kind = "hex", palette = "colorblind")
        # Save plot.
        fig.savefig(SL_folder_path + '/Histogram-' + istr + 'ky.png')
        j += 1
        plt.close()
    
    # Plotting the free reef parameters.
    # Be carefull, makes n_topo * n² plots, with n the numbers of reef param to 
    # inverse.
    
    # Dicos() is used for the axis and figure titles.
    dico = Dicos()
    # Extracts the key of the subdict, and the subdict.
    j = 0
    for key, subdict in dict_df_reef_free.items():
        # Gets the profile number.
        num_profile = int(''.join(filter(str.isdigit, key)))
        # Iteration to extract the first value to plot in chain.samples (x).
        l = j
        j_save = j
        for i in subdict.index :
            # Iteration to extract the second value to plot.
            l = j_save
            # Extracts the description of the first parameter to plot.
            label_i = dico.labels[i]
            # Factor for this parameter.
            factor_i = dico.factors[i]
            for k in subdict.index :
                # Avoids to plot the same parameter.
                if i == k :
                    l += 1
                else :
                    # Description of second parameter.
                    label_k = dico.labels[k]
                    # Its factor
                    factor_k = dico.factors[k]
                    # Creates a dataframe with the two parameters.
                    df = pd.DataFrame({
                        label_i : chain.samples[
                            :, len(df_SL_free) * 2 + j][STP:] * factor_i, 
                        label_k : chain.samples[
                            :, len(df_SL_free) * 2 + l][STP:] * factor_k
                        })
                    # Dataframe to save, with all values.
                    df_save = pd.DataFrame({
                        label_i : chain.samples[
                            :, len(df_SL_free) * 2 + j][:] * factor_i, 
                        label_k : chain.samples[
                            :, len(df_SL_free) * 2 + l][:] * factor_k
                        })
                    # Store the dataframe.
                    df_save.to_pickle(
                        Df_folder_path + '/' + name_files_short[num_profile] + 
                        '_'  + dico.abbrev[i] + '-' + dico.abbrev[k] + '.pkl'
                                 )
                    # Plot.
                    fig = sns.jointplot(data = df, x = label_i, y = label_k, 
                                        kind = "hex", palette = "colorblind")
                    # Save plot.
                    fig.savefig(
                        Folder_path + '/' + name_files_short[num_profile] + 
                        '/Histogram-' + dico.abbrev[i] + '-' + dico.abbrev[k] 
                        + '.png'
                        )
                    l += 1
                    plt.close()
            j += 1
            
    plt.close('all')
        
    
