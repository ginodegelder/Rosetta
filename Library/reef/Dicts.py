from cmcrameri import cm

#********************************************************************************************

class Dicos():
    """ 
    Reads the dictionnaries for plots features 
    for both model runs and the global dataset 
    """

    def __init__(self):
        
        self.root          = './'
        self.path_SLcurves = self.root+'Library/RSLcurves/'
        self.path_zarr     = self.root + 'Outs/Zarr/'
        self.path_anim     = self.root + 'Outs/Anim/'
        self.direOuts = {
            'reef_platform' : self.root + 'Profiles/Outs/reef_platform/',
            'reef'          : self.root + 'Profiles/Outs/reef/',
        }
        
        self.nparams = {
            'reef'          : 4,
            'reef_platform' : 5
        }
        
        self.coltyp = {
            'Barrier'  :'tab:blue', 
            'Fringing' :'tab:cyan', 
            'Other'    :'tab:olive',
            'No'       :'tab:red'
        }
        
        self.colreg = {
            'W Central Pacific':'tab:blue', 
            'North Pacific':'tab:orange', 
            'Central Pacific':'tab:green', 
            'Indian Ocean': 'tab:purple', 
            'Caribbean':'tab:cyan', 
            'Red Sea':'tab:red', 
            'Coral Triangle': 'black'
        }
        

        
        self.rounds = {
            
            'time'             : 0,
            
            'SLstory__Usl'     : 5,
            'SLstory__t_trans' : 0,
            'SLstory__t_stand' : 0,
            
            'grid__slopi'      : 3,
            'grid__dmax'       : 0,
            
            'init__zterr'      : 0,
            'init__lterr'      : 0,
            'init__sloplat'    : 3,
            
            'vertical__u'      : 5,
            
            'construct__Gm'    : 4,
            'construct__hmax'  : 0,
            'hfactor__Dbar'    : 0,
            'hfactor__how'     : 0,
            
            'eros__Ev'         : 3,
            'eros__hwb'        : 0,     

        }
        
        self.test = {
            'Waelbroeck2002b' : 'Waelbroeck2002bWaelbroeck2002bWaelbroeck2002b'
        }
        
        self.SL_files = {
            
            'SLR-15ky'             : self.path_SLcurves+'SLR-15ky.dat',
            'SLR-10ky'             : self.path_SLcurves+'SLR-10ky.dat',
            'Holocene'             : self.path_SLcurves+'Holocene-SL.dat',
            'SLR-20ky'             : self.path_SLcurves+'SLR-20ky.dat',
            'Waelbroeck2002'       : self.path_SLcurves+'Waelbroeck2002.dat',
            'Waelbroeck2002b'      : self.path_SLcurves+'Waelbroeck2002b.dat',
            'Waelbroeck2002-137ky' : self.path_SLcurves+'Waelbroeck2002-137ky.dat',
            'Waeltanja-1500k'      : self.path_SLcurves+'Waeltanja-1500k.dat',

            'Bintanja2008'         : self.path_SLcurves+'Bintanja2008.dat',
            'Bintanja2008-430k'    : self.path_SLcurves+'Bintanja2008-430k.dat',
            'Bintanja2008-1500k'   : self.path_SLcurves+'Bintanja2008-1500k.dat',
            'Bintanja2008-1000k'   : self.path_SLcurves+'Bintanja2008-1000ky.dat',

            'Rohling2009'          : self.path_SLcurves+'Rohling2009.dat',
            'Rohling2009-430k'     : self.path_SLcurves+'Rohling2009-430k.dat',

            'Grant2014'            : self.path_SLcurves+'Grant2014.dat',
            'Grant2014-430k'       : self.path_SLcurves+'Grant2014-430k.dat',

            'Spratt2016'           : self.path_SLcurves+'Spratt2016.dat',
            'Spratt2016-430k'      : self.path_SLcurves+'Spratt2016-430k.dat',

            'Lea2002'              : self.path_SLcurves+'Lea2002.dat',
            'Shakun2015'           : self.path_SLcurves+'Shakun2015.dat',
            'Sinus-asym'           : self.path_SLcurves+'SL-sinus-asym2.dat',
            'Static'               : self.path_SLcurves+'SL-Static.dat'
        }
        
        self.match = {                            # Relating model and global dataset parameters

            'grid__slopi' : 'Slope',
            'vertical__u' : 'U',
        }

        self.scales = {                            # Type of scale to use for the plots depending on the variable

            'time'             : 'linear',
            'Reef_Width'       : 'log',
            'Total_Width'      : 'log',
            'Lag_Width'        : 'log',

            'U'                : 'linear',
            'vertical__u'      : 'linear',

            'Slope'            : 'log',
            'Onsh_Slope'       : 'log',
            'Slope2000'        : 'log',
            'Slope200'         : 'log',
            'grid__slopi'      : 'log',
            'eros__Ev'         : 'log',

            'construct__Gm'    : 'linear',
            'grid__hmax'       : 'linear',
            'construct__delta' : 'linear',
            'hfactor__how'     : 'linear',
            'hfactor__Dbar'    : 'linear',
            'vfactor__Iratio'  : 'linear',
            'vfactor__k'       : 'linear',
            'hfactor__Wbar'    : 'linear',
            
            'SLstory__Usl'     : 'log',
        
        }

        self.cmaps = {                               # Colormaps for z values

            'U'             : 'Spectral',
            'vertical__u'   : 'Spectral',

            'Reef_Width'    : cm.batlow_r,
            'Total_Width'   : cm.lapaz_r,
        }

        self.limits = {                             # Boundaries for model runs and global dataset plots
            
            'time'             : [5e3, 20e3],
            'U'                : [-2, 2],
            'vertical__u'      : [-10e-3, 2e-3],

            'Slope'            : [1, 50],
            'Onsh_Slope'       : [1, 50],
            'Slope2000'        : [1, 50],
            'Slope200'         : [1, 50],
            'grid__slopi'      : [1e-2, 50e-2],

            'construct__Gm'     : [2e-3, 15e-3],
            'grid__hmax'       : [20, 200],
            'hfactor__how'     : [1, 5],
            'hfactor__Dbar'    : [20, 1000],

            'Total_Width'      : [0.09, 30],
            'Reef_Width'       : [0.09, 5],
            
            'vfactor__Iratio'  : [5, 15],
            'vfactor__k'       : [0.05, 0.25],
            'hfactor__Wbar'    : [10, 1000],
            
            'SLstory__Usl'     : [2e-3, 15e-3],
            
            'eros__Ev'         : [50e-3, 1000e-3],

        }

        self.d = {                                 # Shifts for input values, in case of log scale...

            'Total_Width' : 0.1,
            'Reef_Width'  : 0.1,
            'Lag_Width'   : 0.1,

            'Slope'       : 0,
            'Onsh_Slope'  : 0,
            'Slope2000'   : 0,
            'Slope200'    : 0,
            'vfactor__Iratio'  : 0,
            'vfactor__k'       : 0
        }

        self.labels = {                           # Labels for plots axis
            
            'time'               : 'Total duration (ky)',

            'Reef_Width'         : 'Reef width (m)',
            'Total_Width'        : 'Total width (km)',
            'Lag_Width'          : 'Lagoon width (m)',

            'U'                  : 'Vertical rate (mm/y)',
            'vertical__u'        : 'Vertical rate (mm/y)',

            'Slope'              : 'Substrate slope (%)',
            'Onsh_Slope'         : 'Onshore Slope (%)',
            'Slope2000'          : 'Slope from distance to -2000m deep (%)',
            'Slope200'           : 'Slope from distance to -200m deep (%)',
            'grid__slopi'        : 'Initial slope (%)',

            'construct__Gm'      : 'Reef growth rate (mm/y)',
            'grid__dmax'         : 'Grid for erosion (m)',
            'construct__hmax'    : 'Max water height for growth (m)',
            'hfactor__Dbar'      : 'Wave surge (m)',
            'hfactor__how'       : 'Depth of open ocean (m)',
            'eros__Ev'           : 'Erosion rate (mm/y)',
            'eros__hwb'          : 'Wave base depth (m)',
            'grid__slopi'        : 'Initial slope (%)',
            'init__zterr'        : 'Elevation of antecedent terrace (m)',
            'init__lterr'        : 'Length of antecedent terrace (m)',
            
            
            'vfactor__Iratio'    : 'I0 / Ik ratio',
            'vfactor__k'         : 'Extinction coefficient',
            'hfactor__Wbar'      : 'Barrier width (m)',
            
            'SLstory__Usl'       : 'SLR rate (mm/y)',
            
            'Waelbroeck2002'     : 'Waelbroeck et al. (2002)',
            'Bintanja2008'       : 'Bintanja and van de Wal (2008)', 
            'Rohling2009'        : 'Rohling et al. (2009)',
            'Grant2014'          : 'Grant et al. (2014)', 
            'Spratt2016'         : 'Spratt and Lisiecki (2016)', 
            'Lea2002'            : 'Lea et al. (2002)', 
            'Shakun2015'         : 'Shakun et al. (2015)',
            'Holocene'           : ' Holocene',
            'Sinus-asym'         : 'Sinus-asym2',
            
            'Barrier'            : 'Barrier reefs',
            'Fringing'           : 'Fringing reefs',
            'No'                 : 'No reef',
            'Other'              : 'Undefined'
        }

        
        self.factors = {                         # Factors to scale axis to more readable range of values

            'init__sloplat'     : 1e2,
            'init__zterr'       : 1,
            'init__lterr'       : 1,
            
            'time'             : 1e-3,
            
            'SLstory__Usl'     : 1e3,
            'SLstory__t_trans' : 1e-3,
            'SLstory__t_stand' :1e-3,
            
            'grid__dmax'        : 1,
            'grid__slopi'       : 1e2,
                        
            'vertical__u'       : 1e3,
            
            'construct__Gm'      : 1e3,
            'construct__hmax'   : 1,
            'vfactor__Iratio'   : 1,
            'vfactor__k'        : 1,
            'hfactor__how'      : 1,
            'hfactor__Dbar'     : 1,
            'hfactor__Wbar'     : 1,
                        
            'eros__Ev'          : 1e3,
            'eros__hwb'         : 1,
            
            'Total_Width'       : 1e-3,
        }

        self.units = {

            'init__zterr'      : 'm',
            'init__lterr'      : 'm',
            
            'time'             : 'ky',
            
            'vertical__u'      : ' mm/y',

            'grid__slopi'      : '%',

            'construct__Gm'     : ' mm/y',
            'construct__hmax'  : 'm',
            'grid__dmax'       : 'm',
            'construct__delta' : 'm',
            
            'vfactor__Iratio'  : '...',
            'vfactor__k'       : ' ',
            'hfactor__how'     : 'm',
            'hfactor__Dbar'    : 'm',
            'hfactor__Wbar'    : 'm',
            
            'SLstory__Usl'     : 'mm/y',
            
            'eros__hwb'        : 'm',
            'eros__Ev'         : 'mm2/y',
            
        }

        self.titles = {                       

            'time'             : 't max',
            'vertical__u'      : 'U',

            'grid__slopi'      : 'Slope',

            'construct__Gm'     : 'Gm',
            'grid__dmax'       : 'dmax',
            'construct__delta' : 'delta',
            'construct__hmax' : 'hmax',

            'Reef_Width'       : 'Reef width (km)',
            'Total_Width'      : 'Total width (km)',
            
            'vfactor__Iratio'  : 'I0 / Ik',
            'vfactor__k'       : 'k',
            'hfactor__how'     : 'how',
            'hfactor__Dbar'    : 'delta',
            'hfactor__Wbar'    : 'Barrier Width',
            
            'SLstory__Usl'     : 'SLR rate',
            
            'eros__Ev'         : 'Eroded volume',
            'eros__hwb'        : 'Wave base',
            'init__zterr'      : 'Zterr',
            'init__lterr'      : 'Lterr'
            
        }
        
        self.abbrev = {                       

            'time'             : 'tmax',
            'vertical__u'      : 'u',

            'grid__slopi'      : 'slopi',
            'grid__dmax'       : 'dmax',
            'construct__hmax'  : 'hmax',
            
            'init__zterr'      : 'Zterr',
            'init__lterr'      : 'lterr',
            'init__sloplat'    : 'sloplt',

            'construct__Gm'    : 'Gm',            
            'vfactor__Iratio'  : 'I0 / Ik',
            'vfactor__k'       : 'k',
            'hfactor__how'     : 'How',
            'hfactor__Dbar'    : 'Dbar',
            
            'eros__Ev'         : 'Ev',
            'eros__hwb'        : 'Hwb',
            
        }
        
