import numpy as np
import xarray as xr 
from Dicts import Dicos
from Dict_models import DicoModels
from os import path
from datetime import datetime as dtime

#********************************************************************************************
#********************************************************************************************

def CheckSimuProd(ds_in, simu=True, view=False):
        
    # Variables initialisation
    mode = 'w'                            # Mode to write zarr storage
    dm   = DicoModels()
    
    # Check if simulation already exists, runs it if not
    if CheckSimu(ds_in, simu=simu, view=view):
        ds = xr.open_zarr(ZarrName(ds_in))

        # Check if coral growth is recorded
        if ('construct__dG' in list(ds.variables)) == False:
            print('Rerun')
            mode = 'a'
            with dm.models['reef']:
                ds = ds_in.xsimlab.run()

        # Is post-process already done ?
        if ('AccRate_mean' in list(ds.variables)) == False:
            mode = 'a'
            print('Post-proc')
            AccretionRate(ds)
            ds.to_zarr(store = ZarrName(ds_in), mode='a')
            
        return True, ds
    else:
        return False, ds_in
    
#********************************************************************************************s

def WidGraph(param_name, params, ds_in0, simu=True, view=False):
    
    """ 
    Computes reef width and total width in model simulations
    for a given range on a given parameter 

    Parameters:
    -----------
    
    param_name : str
        Name of the parameter as in model input dataset
    params  : array_like
        Array of parameter values to use
    ds_in0 : xarray dataset
        Input dataset to modify with parameter values

    model : xsimlab model
        Model to use for the simulations
        
    **kwargs:
        - simu: boolean, optional
            If True, runs the simulation if it doesn't already exists
            False by default
        - view: boolean, optional
            If True, prints the name of each simulation 
            Useful in case of corrupted zarr storage
            False by default
            
    Returns:
    --------
    
    reefwid: 1d-array 
        Array of reef widths corresponding to parameters values.
    totwid: 1d-array
        Array of total widths corresponding to parameters values.
    lagoon: 1d-array
        Array coding for the reef type: 
        0: Fringing reefs
        1: Barrier reefs
        -1: No simulation
    """
        
    # Variables initialisation
    totwid  = np.zeros(len(params))
    reefwid = totwid.copy()
    lagoon  = np.zeros(len(params))
    dm      = DicoModels()
    
    # Loop on parameter values
    for i in range(len(params)):
        # Update simulation input dataset
        ds_in = ds_in0.xsimlab.update_vars(
            model = dm.models[ds_in0.model_name],
            input_vars={
                param_name : params[i],
            }
        )
        
        # Opening (running ?) simulation
        if CheckSimu(ds_in, simu=simu, view=view):
            ds_out = xr.open_zarr(ZarrName(ds_in))
            
            # Measuring reef and total width
            reef_crest, reefwid[i], totwid[i], typ = ReefWidths(2, ds_out.sealevel__asl.values, ds_out.profile__z.values, ds_out.grid__spacing.values)

            if typ=='bar':
                lagoon[i] = 1
            elif typ == 'fr':
                lagoon[i] = 0
            elif typ == 'no':
                lagoon[i] = 0
        else:
#             print('no sim')
            totwid[i]  = 0.1  
            reefwid[i] = 0.1
            lagoon[i] = -1
        
    return reefwid, totwid, lagoon

#********************************************************************************************s

def WidGraph_t(param_name, params, ds_in0, age_f, simu=True, view=False):
    
    """ 
    Computes reef width and total width in model simulations
    for a given range on a given parameter 

    Parameters:
    -----------
    
    param_name : str
        Name of the parameter as in model input dataset
    params  : array_like
        Array of parameter values to use
    ds_in0 : xarray dataset
        Input dataset to modify with parameter values
    model : xsimlab model
        Model to use for the simulations
        
    **kwargs:
        - simu: boolean, optional
            If True, runs the simulation if it doesn't already exists
            False by default
        - view: boolean, optional
            If True, prints the name of each simulation 
            Useful in case of corrupted zarr storage
            False by default
            
    Returns:
    --------
    
    reefwid: 1d-array 
        Array of reef widths corresponding to parameters values.
    totwid: 1d-array
        Array of total widths corresponding to parameters values.
    lagoon: 1d-array
        Array coding for the reef type: 
        0: Fringing reefs
        1: Barrier reefs
        -1: No simulation
    """
        
    # Variables initialisation
    totwid  = np.zeros(len(params))
    reefwid = totwid.copy()
    lagoon  = np.zeros(len(params))
    dm      = DicoModels()
    
    # Loop on parameter values
    for i in range(len(params)):
        # Update simulation input dataset
        ds_in = ds_in0.xsimlab.update_vars(
            model = dm.models[ds_in0.model_name],
            input_vars={
                param_name : params[i],
            }
        )
        
        # Opening (running ?) simulation
        if CheckSimu(ds_in, simu=simu, view=view):
            ds_out = xr.open_zarr(ZarrName(ds_in))
            
            # Measuring reef and total width
            reef_crest, reefwid[i], totwid[i], typ = ReefWidths(2, ds_out.sealevel__asl.values, ds_out.profile__z.values, ds_out.grid__spacing.values)

            if typ=='bar':
                lagoon[i] = 1
            elif typ == 'fr':
                lagoon[i] = 0
            elif typ == 'no':
                lagoon[i] = 0
        else:
#             print('no sim')
            totwid[i]  = 0.1  
            reefwid[i] = 0.1
            lagoon[i] = -1
        
    return reefwid, totwid, lagoon


#********************************************************************************************

def CheckSimu(ds_in, simu=False, view=False):
    """ 
    Checks if the model run is already done 
    Can run the simulation if not 
    Returns a boolean, so if used with simu=False, 
        it should be used as a condition
    
    Parameters:
    -----------
    
    ds_in : xarray dataset
        Input dataset for the simulation
    model: xsimlab model
        Model used in the simulation
    name_sim: str
        Path and filename for zarr storage
        
    **kwargs:
        - simu: boolean, optional
            Runs the simulation if it doesn't already exists
            False by default
        - view: boolean, optional
            Prints the path and filename of the zarr storage
            False by default
            
    Returns:
    --------
    
    boolean
        True if simulation exists, False otherwise
    """
    
    dm = DicoModels()
    name_sim = ZarrName(ds_in)
    
    if view:
        print(name_sim)
    
    if not path.exists(name_sim):
        if simu:
            t0 = dtime.today()

            print('simu', name_sim)
            
            with dm.models[ds_in.model_name]:
                ds_out = (ds_in   
                      .xsimlab.run(store=ZarrName(ds_in))
                      )
            print('Duration', dtime.today()-t0)
            return True
        else:
            return False     
    else:
        return True
    
#********************************************************************************************

def ModelParameters(model_name):
    """
    Creates a list of useful parameters to use for the simulation

    Parameters:
    -----------

    model_name: str

    Returns:
    --------

    list of str: Useful input parameters for the model
    """

    if model_name == 'reef':
        return ['vertical__u', 'grid__slopi', 'construct__Gm', 'grid__dmax', 'construct__hmax', 'hfactor__Dbar', 'hfactor__how', 'eros__Ev', 'eros__hwb']

    elif model_name == 'reef_platform':
        return ['vertical__u', 'grid__slopi', 'construct__Gm', 'grid__dmax', 'construct__hmax', 'hfactor__Dbar', 'hfactor__how', 'eros__Ev', 'eros__hwb', 'init__zterr', 'init__lterr']
        
    elif model_name == 'reef_eros':
        return ['vertical__u', 'grid__slopi', 'grid__dmax', 'hfactor__how', 'eros__Ev', 'eros__hwb']    

    else:
        print('Arbeit !')
            
#********************************************************************************************

def ZarrName(ds):
    """ 
    Defines the path and filename for zarr storage for a simulation
    for different levels of stored details in the back up of simulations
    
    Parameters:
    -----------
    
    ds : xarray dataset
        Input dataset for the simulation

    Returns:
    --------

    name: str
        Path and filename for zarr storage
    """
    
    # Variables initialisation
    dico = Dicos()
    dt = ds.time[1] - ds.time[0]
    # Path to storage directory
    name = dico.path_zarr
    
    # Storage mode
    if ds.store=='Profile':
        name += 'Profiles/'   
    elif ds.store=='LastProfile':
        name += 'LastProfile/'
    
#     print(ds.SLstory__RSLin.values, type(ds.SLstory__RSLin.values))
    name += ds.model_name+'/'+ds.model_name+'_'+ds.store+'_' + str(ds.SLstory__RSLin.values)
    
    # List of model parameters to use
    params = ModelParameters(ds.model_name)
    print(ds.model_name)
    
    # Loop on parameters to be added to the filename
    for p in params:
        name += '-' +  \
            dico.abbrev[p]+ \
            str(np.round(
                ds[p].values*dico.factors[p], 
                dico.rounds[p]-int(np.log10(dico.factors[p]))
                )
            )
    
    name +='-tmax'+str(round(ds.time[-1].values/1000))+'ky-dt'+str(int(dt))+'y.zarr'
    return name
        
#********************************************************************************************

def ReefWidths(how, e, z, dx):
    """ Measures reef width from how to the same depth shoreward, if any
         Returns the index of the reef crest, the reef width, the total width and the reef type """

    deb = np.argmax(z >= e-how)
    riv = shore(e, z)

    if len(z[deb+1:riv+1]) != 0:
        lagin = np.argmax(z[deb+1:riv] <= e-5)

        if lagin == 0:
            return deb, (shore(e, z)-deb)*dx, (shore(e, z)-deb)*dx, 'fr'
        else:
            return deb, lagin*dx, (shore(e, z)-deb)*dx, 'bar'
    else:
        return deb, 0, 0, 'no'

#********************************************************************************************

def TotalWidth(how, e, z):
    """ Measures the total width of a reef system from how to the shore 
    Profile must be 1D
    SL and how must be scalars """

    return shore(e, z) - np.argmax(z >= e-how)

#********************************************************************************************

def shore(sl, z):
    """
    Finds the index of the shore on a topographic profile for a given elevation of ASL
    Shore is defined as the first node at or below sea level
    Input values must not be xarray DataArray

    Parameters:
    -----------
    
    sl : int, float
        Absolute sea-level elevation 
    z : 1d-array
        Topographic profile
        
    Returns:
    --------
    int: Index of the first node at or below sea level
    """
    
    return np.argmax(z>sl)-1

#********************************************************************************************

def AccretionRate(ds):
    """
    Computes effective reef growth and effective rate on the reef/profile
    _max, _min, _sum values are measured on the whole profile
    _mean values are measured only on the reef (from the location of the 
        first occurrence of hmax to the shore)
    """
    # Variables initialisation
    dt=ds.time[1].values-ds.time[0].values
    ds['construct__dG_mean'] = xr.DataArray(np.zeros_like(ds.time), dims=('time'))
    ds['depot__dS_mean'] = xr.DataArray(np.zeros_like(ds.time), dims=('time'))
    ds['AccRate_mean'] = xr.DataArray(np.zeros_like(ds.time), dims=('time'))
    
    # Measurements along the whole profile    
    ds['construct__dG_max'] = xr.DataArray(np.max(ds.construct__dG, axis=1)/dt, attrs={'units': 'm/y'})
    ds['construct__dG_sum'] = xr.DataArray(np.sum(ds.construct__dG, axis=1)/dt, attrs={'units': 'm/y'})
    ds['depot__dS_max'] = xr.DataArray(np.max(ds.depot__dS, axis=1)/dt, attrs={'units': 'm/y'})
    ds['depot__dS_sum'] = xr.DataArray(np.sum(ds.depot__dS, axis=1)/dt, attrs={'units': 'm/y'})

    
    ds['AccRate_max'] = xr.DataArray(np.zeros_like(ds.time), dims=('time'))
    ds['AccRate_min'] = xr.DataArray(np.zeros_like(ds.time), dims=('time'))
    
    

    acc = np.diff(ds.profile__z, axis=0)-ds.vertical__u.values*dt
    
    dh = ds.sealevel__asl - ds.profile__z
    dh_cond = (dh <= ds.grid__hmax) & (dh >= 0)
    
    print('Start loop')
    for t in range(1, len(ds.time)):
        if (t % 1000 ==0):
            print(t)
        
        if len(dh[t, dh_cond[t]]) > 0:
            ds.construct__dG_mean[t] = ds.construct__dG[t, dh_cond[t]].mean()
            ds.depot__dS_mean[t] = ds.depot__dS[t, dh_cond[t]].mean()
            ds.AccRate_mean[t] = acc[t-1, dh_cond[t]].mean()
            
        else:
            ds.construct__dG_mean[t] = 0
            ds.depot__dS_mean[t] = 0
            ds.AccRate_mean[t] = 0
                
    ds['construct__dG_mean'] = ds.construct__dG_mean / dt
    ds['depot__dS_mean'] = ds.depot__dS_mean / dt
    ds['AccRate_mean'] = ds.AccRate_mean /dt
    
    ds.AccRate_max[1:] = acc.max(axis=1)/dt
    ds.AccRate_min[1:] = np.min(acc, axis=1)/dt

#********************************************************************************************

def CheckLagoon(ds, bar_height, deb):
    """
    Checks if the modern reef has a lagoon
    
    Parameters:
    -----------
    
    ds: xarray dataset
        Input dataset for the simulation
    bar_height: float
        Minimum depth for a significant lagoon

    Returns:
    --------
    
    lag: boolean
        Presence (True) or absence (False) of lagoon
    crest: int
        Index of the crest location
    lagmin: int
        Index of the location of the most oceanward deepest point of th lagoon
    """
    
    # Variables initialisation
    slop = np.round(ds.profile__z[-1, :-1] - ds.profile__z[-1, 1:], 8)
    lag=False
    crest=0
    lagmin=0
    
    # Checking for a lagoon
    if np.any(slop<0):
        first_neg_slop=np.argmax(slop<=0)
        lagmin = np.where(z==np.min(z[first_neg_slop:]))[0][0]
        
        # Finding crest location
        if first_neg_slop!=0:
            crest=np.where(z==np.max(z[:first_neg_slop]))[0][0]
        else:
            print('WTF ?')
            
        # Checking if the lagoon depth is significant
        if z[crest]-z[lagmin]>bar_height:
            lag=True
            
    return lag, crest, lagmin
    
#********************************************************************************************
