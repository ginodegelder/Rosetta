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
    
    0 : [None, None, None, 0, None, None, None], # Actual fixed SL node
    6 : [ 5, 7, 0.3, 2, 1, 3.5, 0.3], # First free SL node
    8.5 : [ 7.5, 9, 0.3, -5, -10, -1, 0.5],
    16 : [None, None, None, -79, None, None, None], # Last fixed SL node
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
    # Timestep for the forward REEF model (years). Use 100 if possible, do not 
    # exceed 400.
    "dt" : 400,
    # Format : 'variable_name' : [starting_point, min, max, step_size]
    # For fixed one : 'variable_name' : [starting_point, None, None, None]
    # Do not change the variable name
    # Everything in meters and years
    "reef_params0" : {
        # vertical land motion rate
        'vertical__u': [-0.27e-3, -0.35e-3, -0.20e-3, 0.1e-3],
        # initial slope of the substrate
        'grid__slopi': [6e-2, None, None, None],
        # Water height for wave base
        'eros__hwb': [3, 1, 5, 0.25],
        # Eroded flux (m^3/y)
        'eros__Ev': [80e-3, 20e-3, 500e-3, 20e-3],
        # Coral reef width construction factor
        'hfactor__Dbar': [500, 200, 1800, 25],
        # maximum reef growth rate
        'construct__Gm': [10e-3, 5e-3, 12e-3, 0.5e-3],
        # maximum water height for reef growth
        'construct__hmax': [20, None, None, None],
        # Water height for open ocean
        'hfactor__how': [2, None, None, None],
        # Length of antecedent terrace. Set to 0 to remove platform
        'init__lterr': [0, None, None, None],
        # Elevation of antecedent terrace
        'init__zterr': [-25, None, None, None],
        # Terrace's slope
        'init__sloplat': [0.e-2, None, None, None],
        # Sinus noise wavelength
        'init__wavelength' : [2500, None, None, None],
        # Sinus amplitude
        'init__amplitude' : [10, None, None, None],
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
        # Eroded flux (m^3/y)
        'eros__Ev': [80e-3, 20e-3, 500e-3, 20e-3],
        # Coral reef width construction factor
        'hfactor__Dbar': [1400, 800, 1800, 25],
        # maximum reef growth rate
        'construct__Gm': [10e-3, 5e-3, 12e-3, 0.5e-3],
        },
    # Another dict
    'reef_params2' : {
        # vertical land motion rate
        'vertical__u': [-0.27e-3, -0.35e-3, -0.20e-3, 0.1e-3],
        # Water height for wave base
        'eros__hwb': [3, 1, 5, 0.25],
        # Eroded flux (m^3/y)
        'eros__Ev': [80e-3, 20e-3, 500e-3, 20e-3],
        # Coral reef width construction factor
        'hfactor__Dbar': [180, 100, 1800, 25],
        # maximum reef growth rate
        'construct__Gm': [10e-3, 5e-3, 12e-3, 0.5e-3],
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
    # Timestep for the forward REEF model (years)
    "dt" : 400,
    # Format : 'variable_name' : [starting_point, min, max, step_size]
    # For fixed one : 'variable_name' : [starting_point, None, None, None]
    # Do not change the values if there is 'X' at the beginning of the comment
    # and do not change the variable name.
    # Everything in meters and years.
    "reef_params0" : {
        # vertical land motion rate
        'vertical__u': [-0.27e-3, -0.35e-3, -0.20e-3, 0.01e-3],
        # initial slope of the substrate
        'grid__slopi': [6e-2, None, None, None],
        # Water height for wave base
        'eros__hwb': [3, None, None, None],
        # Eroded flux (m^3/y)
        'eros__Ev': [400e-3, 20e-3, 400e-3, 30e-3],
        },
# =============================================================================
#   Put here the sub-dicts with only the free parameters for other topo profile
# =============================================================================
    # Second dict
    'reef_params1' : {
        # vertical land motion rate
        'vertical__u': [-0.3e-3, -0.5e-3, -0.1e-3, 0.02e-3],
        # Eroded flux (m^3/y)
        'eros__Ev': [200e-3, 50e-3, 500e-3, 40e-3],
        },
    # Another dict
    'reef_params2' : {
        # vertical land motion rate
        'vertical__u': [-0.23e-3, -0.5e-3, -0.1e-3, 0.02e-3],
        # Eroded flux (m^3/y)
        'eros__Ev': [200e-3, 50e-3, 500e-3, 40e-3],
        }
    }


# =============================================================================
# Parameters for the inversion
# =============================================================================

inversion_params = {
    # Restart from a preexistant. 
    # Put the path of the model 'Out' folder or None.
    'restart' : None,
    # Number of simulations
    'n_samples' : 10, 
    # Tune step size until iteration
    'n_tune' : 1000, 
    # Changes step size every n simu
    'tune_interval' : 500,
    # Starting point for plotting (inferior to n_samples !)
    'stp' : 0,  
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




