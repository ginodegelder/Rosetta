# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:38:23 2023

@author: Yannick Boucharat
"""

# =============================================================================
# Choose your type of model
# =============================================================================

# If construction = True, the model will construct corals. If False, the model
# will be only erosive. Two sets of parameters are disponible for those two 
# types of model.

construction = True

# =============================================================================
# Create your SL curve 
# =============================================================================

sea_level = {
    # Format : t_start : [t_min, t_max, step_t, e_start, e_min, e_max, step_e]
    # For fixed values : t_start : [None, None, None, e, None, None, None]
    # For free nodes, you have to specify every values
    # You can't fix t or e and vary the other value
    # The older value has to be fixed 
    # Time in ky, elevation in meters
    
    0 : [None, None, None, 0, None, None, None],
    6 : [ 5, 7, 0.3, 2, 1, 3.5, 0.3],
    8.5 : [ 7.5, 9, 0.3, -5, -10, -1, 0.5],
    11 : [9.75, 11.75, 0.3, -25, -30, -20, 0.5],
    13.5 : [12.5, 14.5, 0.3, -50, -52.5, -37.5, 0.5],
    15 : [None, None, None, -70, None, None, None],
    16 : [None, None, None, -79, None, None, None],
}


# =============================================================================
# Parameters for the reef construction (construction = True)
# =============================================================================

# For multiple topo inversion, first dict with all the parameters. The other
# ones only contain the free parameters, with the same format.
# The other dictionaries will be sub-dictionnaries in the nesting dictionnary
# "reef_params".
# The order of the sub-dictionnaries correspond to the order of the topographic
# profile files in the Topo_obs directory, in alphabetic order.
# eg : first_subdict --> profileA ; second --> profileB ...

construction_params = {
    # Format : 'variable_name' : [starting_point, min, max, step_size]
    # For fixed one : 'variable_name' : [starting_point, None, None, None]
    # Do not change the values if there is 'X' at the beginning of the comment
    # and do not change the variable name
    # Everything in meters and years
    "reef_params0" : {
        # vertical land motion rate
        'vertical__u': [-0.27e-3, -0.35e-3, -0.20e-3, 0.1e-3],
        # X filename for RSL reconstruction
        'SLstory__RSLin': ['SL', None, None,None],
        # initial slope of the substrate
        'grid__slopi': [6e-2, None, None, None],
        # Water height for wave base
        'eros__hwb': [3, 1, 5, 0.25],
        # Eroded volume
        'eros__Ev': [80e-3, 20e-3, 500e-3, 20e-3],
        # Delta...
        'hfactor__Dbar': [500, 200, 1800, 25],
        # maximum reef growth rate
        'construct__Gm': [10e-3, 5e-3, 12e-3, 0.5e-3],
        # maximum water height for reef growth
        'construct__hmax': [20, None, None, None],
        # X maximum water height for reef growth
        'grid__dmax': [100, None, None, None],
        # X uniform spacing
        'grid__spacing': [1, None, None, None],
        # Water height for open ocean
        'hfactor__how': [2, None, None, None],
        # Coefficient for erosion efficiency, sea-bed
        'eros__beta1': [0.1, None, None, None],
        # Coefficient for erosion efficiency, cliff retreat
        'eros__beta2': [1, None, None, None],
        # Height of notch for volume eroded during cliff retreat
        'eros__hnotch': [1, None, None, None],
        # ---
        'depot__repos': [15e-2, None, None, None],
        # Elevation of antecedent terrace
        'init__zterr': [-25, -35, -20, 1],
        # Length of antecedent terrace
        'init__lterr': [10000, None, None, None],
        # Terrace's slope
        'init__sloplat': 0.e-2,
        # Sinus noise wavelength
        'init__wavelength' : 2500,
        # Sinus amplitude
        'init__amplitude' : 10,
        },
# =============================================================================
#   Put here the sub-dicts with only the free parameters for other topo profile
# =============================================================================
    # Second dict
    'reef_params1' : { 
        # vertical land motion rate
        'vertical__u': [-0.27e-3, -0.35e-3, -0.20e-3, 0.1e-3],
        # Water height for wave base
        'eros__hwb': [3, 1, 5, 0.25],
        # Eroded volume
        'eros__Ev': [80e-3, 20e-3, 500e-3, 20e-3],
        # Delta...
        'hfactor__Dbar': [1400, 800, 1800, 25],
        # maximum reef growth rate
        'construct__Gm': [10e-3, 5e-3, 12e-3, 0.5e-3],
        # Elevation of antecedent terrace
        'init__zterr': [-25, -35, -20, 1],
        },
    # Another dict
    'reef_params2' : {
        # vertical land motion rate
        'vertical__u': [-0.27e-3, -0.35e-3, -0.20e-3, 0.1e-3],
        # Water height for wave base
        'eros__hwb': [3, 1, 5, 0.25],
        # Eroded volume
        'eros__Ev': [80e-3, 20e-3, 500e-3, 20e-3],
        # Delta...
        'hfactor__Dbar': [180, 100, 1800, 25],
        # maximum reef growth rate
        'construct__Gm': [10e-3, 5e-3, 12e-3, 0.5e-3],
        # Elevation of antecedent terrace
        'init__zterr': [-25, -35, -20, 1],
        }
    }

# =============================================================================
# Parameters for the erosive model (construction = False)
# =============================================================================

# For multiple topo inversion, first dict with all the parameters. The other
# ones only contain the free parameters, with the same format.
# The other dictionnaries will be sub-dictionnaries in the nesting dictionnary
# "reef_params".

eros_params = {
    # Format : 'variable_name' : [starting_point, min, max, step_size]
    # For fixed one : 'variable_name' : [starting_point, None, None, None]
    # Do not change the values if there is 'X' at the beginning of the comment
    # and do not change the variable name.
    # Everything in meters and years.
    "reef_params0" : {
        # vertical land motion rate
        'vertical__u': [-0.27e-3, -0.35e-3, -0.20e-3, 0.01e-3],
        # X filename for RSL reconstruction
        'SLstory__RSLin': ['SL', None, None,None],
        # initial slope of the substrate
        'grid__slopi': [6e-2, None, None, None],
        # Water height for wave base
        'eros__hwb': [3, None, None, None],
        # Eroded volume
        'eros__Ev': [400e-3, 20e-3, 400e-3, 30e-3],
        # X maximum water height for reef growth
        'grid__dmax': [100, None, None, None],
        # X uniform spacing
        'grid__spacing': [1, None, None, None],
        # Coefficient for erosion efficiency, sea-bed
        'eros__beta1': [0.1, None, None, None],
        # Coefficient for erosion efficiency, cliff retreat
        'eros__beta2': [1, None, None, None],
        # Height of notch for volume eroded during cliff retreat
        'eros__hnotch': [1, None, None, None],
        },
# =============================================================================
#   Put here the sub-dicts with only the free parameters for other topo profile
# =============================================================================
    # Second dict
    'reef_params1' : {
        # vertical land motion rate
        'vertical__u': [-0.3e-3, -0.5e-3, -0.1e-3, 0.02e-3],
        # Eroded volume
        'eros__Ev': [200e-3, 50e-3, 500e-3, 40e-3],
        },
    # Another dict
    'reef_params2' : {
        # vertical land motion rate
        'vertical__u': [-0.23e-3, -0.5e-3, -0.1e-3, 0.02e-3],
        # Eroded volume
        'eros__Ev': [200e-3, 50e-3, 500e-3, 40e-3],
        }
    }


# =============================================================================
# Parameters for the inversion
# =============================================================================

inversion_params = {
    # Number of simulations
    'n_samples' : 10, 
    # Factor to change the step size after every tune_interval simu
    'n_tune' : 1000, 
    # Changes step size every n simu
    'tune_interval' : 500,
    # Starting point for plotting (inferior to n_samples !)
    'stp' : 0,  
    # dx for reef inversion, do not change it
    'dx_reef' : 1,  
    # Amplitude in m, related to uncertainty in cliff height
    'sigma' : 5,  
    # Correlation length (multiple of ipstep), related to uncertainty in terrace width
    'corr_l' : 3, 
    # Every ipstep on x axis, check dz : difference btw simu and observation
    'ipstep' : 100,  
    }


# =============================================================================
# Observed topography to compare with simulation
# =============================================================================

topo_obs = {
    # Path of the text file with x and elevation values of the observed topo
    'topo_path' : './Topo_obs/'
    }




