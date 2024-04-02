import numpy as np
import xsimlab as xs
from tools import shore, readfile, apply_mask_nb
from math import log10, exp, floor
from Dicts import Dicos
from functools import reduce
from PythranTools import Volume, vertical_eros, fill_nodes, limits



#*********************************************************************
#*********************************************************************

@xs.process
class SeaLevel:
    """ 
    Defines which SL scenario will be used 
    """
    
    asl = xs.variable(intent='out', description='Absolute Sea Level at each time step', attrs={'units':'m'})
    asl_vars = xs.group("asl_vars")
        
    def run_step(self):

        self.asl = np.sum(self.asl_vars, axis=0)
        
#*********************************************************************

@xs.process
class SLFile:
    """
    Reads SL variations from a file and 
    interpolates it according to the time step
    """
        
    RSLin=xs.variable(intent='in', description='filename for RSL reconstruction')
    asl_in = xs.variable(dims='time[-1]', intent='out', description='interpolated elevations of paleoSL', attrs={'units':'m'})
    asl_file = xs.variable(dims='time', intent='out', groups='asl_vars')

    def initialize(self):
        
        dico = Dicos()
        
        age_in, self.asl_in = readfile(dico.path_SLcurves+self.RSLin)
    
        ###age_in, self.asl_in = readfile(dico.SL_files[self.RSLin])
        self.t_in = np.arange(age_in[-1], age_in[0]+1)*1000
    
    @xs.runtime(args=('step_start'))
    def run_step(self, t):
    
        if t % 100000 == 0:
           print('t', t/1000)
        # Interpolates SL for the timestep
        self.asl_file = np.interp(t, self.t_in, self.asl_in)  
        
#*********************************************************************

@xs.process
class SLRise:
    """ 
    Creates a Sea Level Rise scenario for a given rate and duration 
    """
    
    Usl = xs.variable(intent='in', description='Rate of SL rise', attrs={'units' : 'm/y'})
    eSLR = xs.variable(dims='time', intent='out', groups='asl_vars')
    
    @xs.runtime(args=('sim_end'))
    def initialize(self, tmax):
        
        self.eSLR = -tmax*self.Usl
        
    @xs.runtime(args=('step_delta', 'step_start'))
    def run_step(self, dt, t):

        self.eSLR += self.Usl*dt
        
#*********************************************************************

@xs.process
class VerticalDisp:
    """
    Uniform and constant vertical displacement
    See later for scenarios...
    """

    u = xs.variable(intent="in", description="vertical land motion rate", attrs={"units":"m/y"})
    du=xs.variable(dims="x", intent="out", groups="z_vars")

    @xs.runtime(args=('step_delta'))
    def run_step(self, dt):
        
        # Vertical displacement for the time step
        self.du=self.u*dt
        
#*********************************************************************

@xs.process   
class UniformGrid1D:
    """
    Creates a 1-dimensional, equally spaced grid
    """
    
    dmax = xs.variable(description="maximum water height for acting processes (hmax or hwb)", attrs={'unit':'m'})
    spacing = xs.variable(description="uniform spacing", static=True, attrs={'unit':'m'}, default = 1)
    slopi = xs.variable(description="initial slope of the substrate", static=True, attrs={'unit':'%'})    

    length = xs.variable(intent='out', description='Length of profile', attrs={'units':'m'})    
    x = xs.index(dims="x")
    
    u = xs.foreign(VerticalDisp, 'u')
 
    @xs.runtime(args = ('sim_end'))
    def initialize(self, tmax):
        
        # Defines grid length and indices, to be improved...
        self.length = int((self.dmax + 900 + abs(self.u) * tmax) / self.slopi)+6000
        if self.slopi > 10e-2:
            self.length += 1700
        self.x = np.arange(0., self.length, self.spacing)
        
#*********************************************************************

@xs.process
class ProfileZ:
    """
    Computes the evolution of the elevation profile (z)
    """
    
    xmin = xs.variable(intent="inout", description='starting index for final plot')
    xmax = xs.variable(intent="inout", description='ending index for final plot')
    z_vars = xs.group("z_vars")
    z = xs.variable(dims="x", intent="inout", description="elevation z", attrs={"units":"m"})
#     dh = xs.variable(dims='x', intent='out')
    
    u = xs.foreign(VerticalDisp, 'u')
    asl = xs.foreign(SeaLevel, 'asl')
    x = xs.foreign(UniformGrid1D, 'x')
    
    @xs.runtime()
    def run_step(self):
        
        # Runs the processes associated with profile evolution
        self._delta_z = np.sum(self.z_vars, axis=0)

    @xs.runtime(args=('step_delta', 'sim_start', 'sim_end'))
    def finalize_step(self, dt, t, tmax):
        
        # Modifying topographic profile
        self.z += self._delta_z
                  
        # Actualising profile boundaries        
        condition_x = self.x[np.abs(self._delta_z - self.u*dt) >= 1e-4]
        if len(condition_x) != 0:
            
            self.xmin = min(
                self.xmin, 
                np.min(condition_x))
            
            self.xmax = max(
                    self.xmax,
                    shore(self.asl, self.z)+50)
            
#*********************************************************************

@xs.process
class InitTopo:
    """Initializes a linear initial profile for a given slope"""

    xmin = xs.foreign(ProfileZ, 'xmin', intent='out')
    xmax = xs.foreign(ProfileZ, 'xmax', intent='out')
    x = xs.foreign(UniformGrid1D, "x")
    z = xs.foreign(ProfileZ, "z", intent="out")
    dmax = xs.foreign(UniformGrid1D, 'dmax')
    slopi = xs.foreign(UniformGrid1D, 'slopi')
    u = xs.foreign(VerticalDisp, 'u')
    
    @xs.runtime(args=('sim_end'))
    def initialize(self, tmax):
        
        # Initialisation of profile boundaries
        self.xmax=self.x[0]
        self.xmin = self.x[-1]
        
        # Initial profile
        if int(self.u*1e5) > 0:
            if self.slopi>=10e-2:
                shift = 750
            else:
                shift = 400
            self.z = self.slopi * self.x - (self.dmax+abs(self.u*tmax)+shift)
        else:
            if self.slopi>=10e-2:
                shift = 1000
            else:
                shift = 500
            self.z = self.slopi * self.x -(shift+self.dmax)

#*********************************************************************

@xs.process
class InitTopoTerr:
    """Initializes a customized initial profile"""

    zterr = xs.variable(description="Elevation of antecedent terrace", attrs={"units":"m"})
    lterr = xs.variable(description="Length of antecedent terrace", attrs={"units":"m"})
    sloplat = xs.variable(description = "Terrace's slope", attrs={"units":"%"})
    wavelength = xs.variable(description = "Sinus noise wavelength", attrs={"units":"m"})
    amplitude = xs.variable(description = "Ampliyude of terrace's noise elevation", attrs={"units":"m"})
    
    xmin = xs.foreign(ProfileZ, 'xmin', intent='out')
    xmax = xs.foreign(ProfileZ, 'xmax', intent='out')
    x = xs.foreign(UniformGrid1D, 'x')
    z = xs.foreign(ProfileZ, 'z', intent='out')
    dmax = xs.foreign(UniformGrid1D, 'dmax')
    slopi = xs.foreign(UniformGrid1D, 'slopi')
    u = xs.foreign(VerticalDisp, 'u')
    
    @xs.runtime(args=('sim_end'))
    def initialize(self, tmax):

        self.xmin=self.x[-1]
        self.xmax=self.x[0]
        
        if self.u > 0:
            self.z = self.slopi * self.x - (self.dmax+abs(self.u*tmax)+500)
        else:
            self.z = self.slopi * self.x -(700+self.dmax)
            
        # Extracts platform's x bounds
        end = np.argmax(self.z >= self.zterr)
        start = max(50, end-int(self.lterr))
        # self.z[start:end] = self.zterr

        t=np.arange(self.lterr) # Platform's x values
        U=self.amplitude*np.sin(2*np.pi*t/self.wavelength) # Platform's sinusoid values
        # Terracing
        self.z[start:end] = self.zterr + np.arange(end-start) * self.sloplat - (end-start) * self.sloplat + U
        # Fill the potential cliff facing the coast
        if U[-1] > 0:
            self.z[end-1 + np.where(self.z[end-1:]<self.z[end-1])] = self.z[end-1]
            
        
#*********************************************************************

@xs.process
class InitTopoTerr2:
    """Initializes a customized initial profile"""

    zterr = xs.variable(description="Elevation of antecedent terrace", attrs={"units":"m"})
    lterr = xs.variable(description="Length of antecedent terrace", attrs={"units":"m"})
    sloplat = xs.variable(description = "Slope of the platform", attrs = {'units':'%'})
    
    xmin = xs.foreign(ProfileZ, 'xmin', intent='out')
    xmax = xs.foreign(ProfileZ, 'xmax', intent='out')
    x = xs.foreign(UniformGrid1D, 'x')
    z = xs.foreign(ProfileZ, 'z', intent='out')
    dmax = xs.foreign(UniformGrid1D, 'dmax')
    slopi = xs.foreign(UniformGrid1D, 'slopi')
    u = xs.foreign(VerticalDisp, 'u')
    
    @xs.runtime(args=('sim_end'))
    def initialize(self, tmax):

        self.xmin=self.x[-1]
        self.xmax=self.x[0]
        
        # Designs a linear slope
        if self.u > 0:
            self.z = self.slopi * self.x - (self.dmax+abs(self.u*tmax)+500)
        else:
            self.z = self.slopi * self.x -(700+self.dmax)
            
        # Terracing
        fin = np.argmax(self.z >= self.zterr)
        deb = max(50, fin-int(self.lterr))
        self.z[deb:fin] = self.zterr + np.arange(fin-deb) * self.sloplat
        
#*********************************************************************

@xs.process
class WaterHeight:
    """ Computes water height and finds shore """
    
    dh = xs.variable(dims = 'x', intent='out', attrs={'units':'m'})
    riv = xs.variable(intent='out')
    
    z = xs.foreign(ProfileZ, 'z')
    asl = xs.foreign(SeaLevel, 'asl')
    
    def run_step(self):
        
        self.dh = self.asl - self.z
        self.riv = shore(self.asl, self.z)

#*********************************************************************

@xs.process
class Construction:
    """Reefal construction based on input maximum ree growth rate Gm
    Modulated by factors """
    
    # Getting input parameters
    Gm = xs.variable(intent="in", description="maximum reef growth rate", attrs={'units': 'm/y'})
    Gf_vars = xs.group("Gf_vars")
    dG = xs.variable(dims="x", intent="out", groups="z_vars", attrs={'units': 'm'})
    hmax = xs.variable(intent="in", description="maximum water height for reef growth", attrs={'units': 'm'})

    # Getting foreign variables
    x = xs.foreign(UniformGrid1D, 'x', intent='in')
    dh = xs.foreign(WaterHeight, 'dh')
    riv = xs.foreign(WaterHeight, 'riv')

    @xs.runtime(args=("step_delta"))
    def run_step(self, dt):

        self.dG = self.Gm * dt * reduce(lambda x, y: x * y, self.Gf_vars)
        
        # Limitation of reef growth by sea level        
        mask = (self.x <= self.x[self.riv]) & (self.dG > self.dh-0.1)        
        self.dG[mask] = self.dh[mask]-0.1 # Number 2
        self.dG[self.dG<0]=0.
        
#*********************************************************************

@xs.process
class MyVerticalFactor:
    """ Computes my vertical factor """
    
    Vf = xs.variable(dims="x", intent='out', description='My vertical factor', groups="Gf_vars")
    
    x = xs.foreign(UniformGrid1D, "x", intent='in')
    hmax = xs.foreign(Construction, 'hmax')
    dh = xs.foreign(WaterHeight, 'dh')
    riv = xs.foreign(WaterHeight, 'riv')
    
    def run_step(self):

        self.Vf = np.zeros(self.x.size)
        mask = (self.dh <= self.hmax) & (self.x <= self.x[self.riv])
        self.Vf[mask] = (1. + np.cos(np.pi * self.dh[mask] / self.hmax)) / 2

#*********************************************************************

@xs.process
class MyHorizontalFactor:
    """ Computes my horizontal factor """
    
    how = xs.variable(intent='in', description='Water height for open ocean', attrs={'units': 'm'})
    Dbar = xs.variable(intent='in', description='Delta...', attrs={'units': 'm'})
    
    xow = xs.variable(intent='out', description='Location of open ocean')
    Hf = xs.variable(dims="x", intent='out', description='My horizontal factor', groups="Gf_vars")
    
    x = xs.foreign(UniformGrid1D, 'x', intent='in')
    dh = xs.foreign(WaterHeight, 'dh')
    riv = xs.foreign(WaterHeight, 'riv')
    
    def run_step(self):
        
        # Distance to the open water
        ow=np.argmax(self.how>=self.dh)
        self.xow=self.x[ow]+np.arctanh(0.99)*self.Dbar

        # Horizontal factor
        self.Hf = np.zeros(self.x.size)
        
        mask = self.x <= self.x[self.riv]
        self.Hf[mask] = (np.tanh((self.xow-self.x[mask])/self.Dbar)+1)/2

#*********************************************************************
@xs.process
class ErosiveMemory:
    
    cr_mem = xs.variable(description = 'Memory profile for cliff retreat', dims='x', intent='out')
    
    x = xs.foreign(UniformGrid1D, 'x')
    dG = xs.foreign(Construction, 'dG')
    
    def initialize(self):
        
        self.cr_mem = np.zeros(len(self.x))   
        
    def finalize_step(self):
        
        # Sets erosive memory to 0 if the reef grew on it
        self.cr_mem[self.dG>0]=0

#*********************************************************************
@xs.process
class ErosiveMemory_ErosOnly:
    
    cr_mem = xs.variable(description = 'Memory profile for cliff retreat', dims='x', intent='out')
    
    x = xs.foreign(UniformGrid1D, 'x')
    
    def initialize(self):
        
        self.cr_mem = np.zeros(len(self.x))   
        
#*********************************************************************

@xs.process
class ErosionConst:

    Ev = xs.variable(intent='in', description = 'Eroded volume', attrs={'units':'m2/y'})
    hwb = xs.variable(intent = 'in', description = 'Water height for wave base', attrs={'units':'m'})
    dE = xs.variable(dims='x', intent='out', groups='z_vars', attrs={'units': 'm'})
    beta1 = xs.variable(description = 'Coefficient for erosion efficiency, sea-bed')
    beta2 = xs.variable(description = 'Coefficient for erosion efficiency, cliff retreat')
    hnotch = xs.variable(description = 'Height of notch for volume eroded during cliff retreat', attrs={'units':'m'})
    start_eros = xs.variable(intent='out')

    asl = xs.foreign(SeaLevel, 'asl', intent='in')
    x = xs.foreign(UniformGrid1D, 'x')
    z = xs.foreign(ProfileZ, 'z', intent='in')
    dh = xs.foreign(WaterHeight, 'dh')
    riv = xs.foreign(WaterHeight, 'riv')
    cr_mem = xs.foreign(ErosiveMemory, 'cr_mem')
    length = xs.foreign(UniformGrid1D, 'length')
    
    @xs.runtime(args=('step_delta', 'step_start'))
    def run_step(self, dt, t):  
                
        # Variables initialisation
        Vrest=self.Ev*dt
        tmp=np.zeros(3)        
        eps = Vrest/1000
        self.dE = np.zeros(self.x.size)
        test_fin = 0
        box = self.hnotch*self.beta2
        hwbb = self.hwb/4
        
        # Starting point for vertical sea-bed erosion
        self.start_eros = np.argmax(self.dh<=self.hwb)
              
        # Vertical sea-bed erosion
        i=self.start_eros
        test = np.empty(len(self.dh))
        test[i:self.riv+1] = np.exp(-self.dh[i:self.riv+1]/hwbb)*self.beta1
        Vrest, self.dE = vertical_eros(i, self.riv, Vrest, eps,
                                            self.dh, self.hwb, self.z,
                                            self.dE, test)
        
        # Cliff retreat
        if Vrest > 0:
            # Starts at first terrestrial node
            j=self.riv+1
            
            while (j < len(self.z)) & (Vrest > 0):
                
                # If it isn't an opened fossil lagoon
                if self.z[j]>self.asl-0.1:

                    # Temporary profile for volume
                    tmp=self.z[j-1:j+2]+self.dE[j-1:j+2]
                    tmp[1]=self.asl-0.1
                    
                    # Volume to be eroded                
                    minvol=min(abs(Volume(self.z[j-1:j+2]+self.dE[j-1:j+2], tmp).sum()), box)
                    
                    # Is there enough wave energy left ?
                    if Vrest+self.cr_mem[j] >= minvol:
                        self.dE[j] = tmp[1] - self.z[j]
                        Vrest = Vrest - (minvol-self.cr_mem[j])
                        self.cr_mem[j] = 0
                            
                        # Computing associated sea-bed erosion
                        tmp2=tmp.copy()
                        tmp2[1] = max(tmp[1]-Vrest*exp(-0.1/hwbb)*self.beta1, self.z[self.riv]+self.dE[self.riv])
                        seabed_vol = abs(Volume(tmp, tmp2).sum())
                        
                        # Is there enough wave energy left ?
                        if Vrest >= seabed_vol:
                            self.dE[j] += (tmp2[1]-(self.z[j]+self.dE[j]))
                            Vrest -= seabed_vol
                        else:
                            Vrest = 0.
                            break
                        # Removing eroded volume to erodible volume
                        
                        j += 1

                    else:
                        self.cr_mem[j] += Vrest   
                        Vrest = 0
                        break
                else:
                    j+=1
        
        # if abs(self.dE.sum() - self.cr_mem[j]) < self.Ev*dt/2:
        #     print('New WTF, t', t, 'dE', self.dE.sum(), 'crmem', self.cr_mem[j])
            
#*********************************************************************

@xs.process
class ErosOnly:

    Ev = xs.variable(intent='in', description = 'Eroded volume', attrs={'units':'m2/y'})
    hwb = xs.variable(intent = 'in', description = 'Water height for wave base', attrs={'units':'m'})
    dE = xs.variable(dims='x', intent='out', groups='z_vars', attrs={'units': 'm'})
    beta1 = xs.variable(description = 'Coefficient for erosion efficiency, sea-bed')
    beta2 = xs.variable(description = 'Coefficient for erosion efficiency, cliff retreat')
    hnotch = xs.variable(description = 'Height of notch for volume eroded during cliff retreat', attrs={'units':'m'})
    start_eros = xs.variable(intent='out')

    asl = xs.foreign(SeaLevel, 'asl', intent='in')
    x = xs.foreign(UniformGrid1D, 'x')
    z = xs.foreign(ProfileZ, 'z', intent='in')
    dh = xs.foreign(WaterHeight, 'dh')
    riv = xs.foreign(WaterHeight, 'riv')
    cr_mem = xs.foreign(ErosiveMemory_ErosOnly, 'cr_mem')
    length = xs.foreign(UniformGrid1D, 'length')
    
    @xs.runtime(args=('step_delta', 'step_start'))
    def run_step(self, dt, t):  
                
        # Variables initialisation
        Vrest=self.Ev*dt
        tmp=np.zeros(3)        
        eps = Vrest/1000
        self.dE = np.zeros(self.x.size)
        test_fin = 0
        box = self.hnotch*self.beta2
        hwbb = self.hwb/4
        
        # Starting point for vertical sea-bed erosion
        self.start_eros = np.argmax(self.dh<=self.hwb)
              
        # Vertical sea-bed erosion
        i=self.start_eros
        test = np.empty(len(self.dh))
        test[i:self.riv+1] = np.exp(-self.dh[i:self.riv+1]/hwbb)*self.beta1
        Vrest, self.dE = vertical_eros(i, self.riv, Vrest, eps,
                                            self.dh, self.hwb, self.z,
                                            self.dE, test)
        
        # Cliff retreat
        if Vrest > 0:
            # Starts at first terrestrial node
            j=self.riv+1
            
            while (j < len(self.z)) & (Vrest > 0):
                
                # If it isn't an opened fossil lagoon
                if self.z[j]>self.asl-0.1:

                    # Temporary profile for volume
                    tmp=self.z[j-1:j+2]+self.dE[j-1:j+2]
                    tmp[1]=self.asl-0.1
                    
                    # Volume to be eroded                
                    minvol=min(abs(Volume(self.z[j-1:j+2]+self.dE[j-1:j+2], tmp).sum()), box)
                    
                    # Is there enough wave energy left ?
                    if Vrest+self.cr_mem[j] >= minvol:
                        self.dE[j] = tmp[1] - self.z[j]
                        Vrest = Vrest - (minvol-self.cr_mem[j])
                        self.cr_mem[j] = 0
                            
                        # Computing associated sea-bed erosion
                        tmp2=tmp.copy()
                        tmp2[1] = max(tmp[1]-Vrest*exp(-0.1/hwbb)*self.beta1, self.z[self.riv]+self.dE[self.riv])
                        seabed_vol = abs(Volume(tmp, tmp2).sum())
                        
                        # Is there enough wave energy left ?
                        if Vrest >= seabed_vol:
                            self.dE[j] += (tmp2[1]-(self.z[j]+self.dE[j]))
                            Vrest -= seabed_vol
                        else:
                            Vrest = 0.
                            break
                        # Removing eroded volume to erodible volume
                        
                        j += 1

                    else:
                        self.cr_mem[j] += Vrest   
                        Vrest = 0
                        break
                else:
                    j+=1
        
        # if abs(self.dE.sum() - self.cr_mem[j]) < self.Ev*dt/2:
        #     print('New WTF, t', t, 'dE', self.dE.sum(), 'crmem', self.cr_mem[j])
            

#*********************************************************************
#*********************************************************************

#*********************************************************************
#*********************************************************************

@xs.process()
class SedimClastics():

    """Deposits clastic sediments from wave erosion and cliff collapse
        - One way from the depth of wave base to the shore
    """
    
    dS = xs.variable(dims='x', intent='out', groups='z_vars', attrs={'units': 'm'})
    Vsed = xs.variable(dims='x', intent='out')
    repos = xs.variable (intent = 'in', attrs={'units': '%'})
    
    dG = xs.foreign(Construction, 'dG')
    start_eros = xs.foreign(ErosionConst, 'start_eros')
    z = xs.foreign(ProfileZ, 'z')
    dE = xs.foreign(ErosionConst, 'dE')
    asl = xs.foreign(SeaLevel, 'asl')
    xmin = xs.foreign(ProfileZ, 'xmin')

        
    @xs.runtime(args=('step_start'))
    def run_step(self, t):
        
        # Variables initialisation
        self.dS=np.zeros(len(self.z))
        self.t = t
        self.Vsed=np.empty(len(self.z))
        self.z_tmp = self.z + self.dE + self.dG
        # self.dS_init = np.copy(self.dS)
        # self.z_tmp_init = np.copy(self.z_tmp) #+ self.dS_init
        self.riv = shore(self.asl, self.z_tmp)
        self.len_z_plus1 = len(self.z)+1
                    
        # Computing total eroded volume
        self.Vsed = Volume(self.z, self.z-self.dE)
        # First round of deposition shorewards
        self.SedimLagoonDyn()
        
        self.z_tmp_init = self.z_tmp.copy()
        # if np.any(self.dS<0):
        #     print('MERDE LAGOON !')
            
        # Starting repose deposition from deb (hwb), at least for now...
        self.Vtot = self.Vsed.sum()
        
        slop=np.diff(self.z_tmp)
        start_lay = self.start_eros-1 - np.argmax((slop[self.start_eros-1:0:-1] < self.repos) & (self.z_tmp[self.start_eros-1:0:-1] < self.z_tmp[self.start_eros]))  

        Slop_round = round(slop[1], 2)
        arr_to_fill = np.arange(self.start_eros+1) * self.repos 
        ln_to_fill = len(arr_to_fill) 
        # self.j_save = np.empty(0, dtype = "int")
        while (start_lay > 0) & (self.Vtot > 0):
            
            if slop[start_lay] <= self.repos:
                while (slop[start_lay]<=self.repos) & (self.Vtot > 0):

                    lay_to_fill = arr_to_fill[:ln_to_fill-start_lay]                    
                    self.FillLayer(start_lay, lay_to_fill)
                    start_lay-=1
                    # Get out of the loop if initial slope > repose angle and no more construction/erosion below
                    if Slop_round >= self.repos:
                        if start_lay < self.xmin:
                            # On perd tous les sédiments
                            self.Vtot = 0.
                            break                
                
            else:
                start_lay-=1
                
#         if self.dS.sum() == 0:
#             print('No sed, t', t, 'dE', self.dE.sum())
                
#*********************************************************************

    def FillLayer(self, dep, lay):
        """ Fills a layer of sediment along a repose angle from a starting point (dep)
            Limited by hwb or topography """

        
        # Variables initialisation
        ddS=self.repos                                            # Sediment thickness deposited at each node, to modify to be able to scale dx

        # mask = np.where(self.dS!=0)
        # self.z_tmp[mask] = self.z_tmp_init[mask] + self.dS[mask] 
        
        self.z_tmp = apply_mask_nb(self.z_tmp, self.z_tmp_init, self.dS)      
        
        Fut_z = self.z_tmp[dep] + lay
        end, Fut_z = limits(Fut_z, self.z_tmp, self.len_z_plus1, self.start_eros, dep)

        if end == True:
            return        
        
        cut_z_tmp = self.z_tmp[dep:end+1]

        Vol_layer = Volume(cut_z_tmp, Fut_z).sum()

        # Is there enough sediments ?
        if Vol_layer <= self.Vtot:
            self.dS[dep:end+1] += (Fut_z - cut_z_tmp) 
            self.Vtot -= Vol_layer  
        
        else:
            # Filling the layer node by node until no sediment remains  ## j_save as inout, store in loop
            self.Vtot, self.dS, self.z_tmp = fill_nodes(
                dep, end, self.z_tmp, ddS, self.Vtot, self.dS)
                
#*********************************************************************

    def FillLayer0(self, dep):
        """ Fills a layer of sediment along a repose angle from a starting point (dep)
            Limited by hwb and topography """

        # Variables initialisation
        ddS=self.repos
        last = False
        # Loop on layer's nodes
        for j in range(dep+1, len(self.z)):

            # Checking if future elevation is above topography
            Fut_zj = self.z_tmp[j-1:j+2].copy()
            if len (Fut_zj) ==0:
                print('Why ??? t', self.t, j-1, j+2)
            Fut_zj[1] = Fut_zj[0] + ddS
            if Fut_zj[1] >= self.z_tmp[self.start_eros]:
                Fut_zj[1] = self.z_tmp[self.start_eros]
                last = True
            
            if Fut_zj[1] < self.z_tmp[j]:
                break

            # Is there enough sediments ?
            Vol_node = Volume(self.z_tmp[j-1:j+2], Fut_zj).sum()
            # Yes
            if self.Vtot >= Vol_node: 
                self.dS[j] += (Fut_zj[1] - self.z_tmp[j])
                self.z_tmp[j] = Fut_zj[1]
                self.Vtot -= Vol_node
                
                if np.any(self.dS <0):
                    print('MERDE !', Fut_zj[1] - self.z_tmp[j], Fut_zj[1], self.z_tmp[j])
                if last:
                    return
            # No
            else:    
                self.Vtot = 0.
                return
            
#*********************************************************************
        
    def SedimLagoonDyn(self):

        """ Drops sediments in holes one after another ... 
            First deposition in the whole cycle """
        
        # Computing slope for the lagoon
        slop=self.z_tmp[self.start_eros+1:self.riv+2]-self.z_tmp[self.start_eros:self.riv+1]
        if len(slop)==0:
            return
        # Finding local minima and maxima
        if np.any(slop<0):
            jmins = self.FindHoles(slop)
            jmaxs = self.FindPeaks(slop)
        else:
            return
        
        # Reef crest is defined as as the first peak above hwb
        crest = jmaxs[0]
        self.Vsed[crest]+=self.Vsed[self.start_eros:crest].sum()
        self.Vsed[self.start_eros:crest]=0.

        # Initialize first hole coordinates
        k=0
        jfd = jmins[k]
        ju0 = jmaxs[k]
        ju1 = jmaxs[k+1]
            
        # Depositing sediments
        while self.Vsed.sum()>=0:
            
            # Remplissage du "premier" trou
            spill = self.FillHole(ju0, jfd, ju1)
            
            # Modifying holes coordinates/index
            # Si le dernier a été comblé, on le supprime des tableaux qui sont ajustés en conséquence
            if spill!=0 :
                # Si ce dernier trou est limité par la crete recifale, on passe a la sedimentation au large
                if (spill == crest) and len(jmins) == 1:
                    break
                # Le rivage est le facteur limitant, et oui, ça arrive mais je ne comprends pas encore pourquoi...
                if spill == jmaxs[-1]:
                    print('Is this happening ?')
                    break
                
                # Adjusting jmins and jmaxs accordingly
                tmp=np.empty(len(jmins)-1, dtype='int')
                tmp[0:k] = jmins[0:k]
                tmp[k:] = jmins[k+1:]
                jmins=tmp
                
                tmp = np.empty(len(jmaxs)-1, dtype='int')
                tmp[0:np.argmax(jmaxs==spill)] = jmaxs[0:np.argmax(jmaxs==spill)]
                tmp[np.argmax(jmaxs==spill):] = jmaxs[np.argmax(jmaxs==spill)+1:]
                jmaxs=tmp
                
                # What's the next hole ?
                if ((spill == ju0) & (k != 0)):
                    k-=1
                
            # The last hole was not filled
            else:
                if k != len(jmins)-1:
                    k+=1
                else:
                    break

            # Preparing for next hole, if any
            if len(jmins !=0):
                tmp = self.Vsed[jfd]
                self.Vsed[jfd]=0
                jfd = int(jmins[k])
                self.Vsed[jfd] += tmp
            
                ju0 = int(jmaxs[k])
                ju1 = int(jmaxs[k+1])

            else:      # On a comblé tous les trous entre la côte et la wave base depth et il reste des sédiments
                break
                                
#*********************************************************************

    def FillHole(self, ju0, jfd, ju1):

        """ Fills one hole between two upper boundaries ju0 and ju1 
        For now at least, the profile has to be from the crest to the shore... """

        # All sediments are gathered at the deepest point
        tmp = self.Vsed[ju0:ju1+1].sum()
        self.Vsed[ju0:ju1+1]=0
        self.Vsed[jfd] = tmp

        dh_tmp_0 = self.z_tmp[ju0:ju1+1]+self.dS[ju0:ju1+1]
        # Defining the spilling point
        spill = ju0
        if self.z_tmp[ju0]+self.dS[ju0] > self.z_tmp[ju1]+self.dS[ju1]:
            spill = ju1
            
        # Is there enough sediment to fill the hole ?
        tmp_z = self.FlatHole(ju0, ju1, self.z_tmp[spill])
        Vol_hole = Volume(dh_tmp_0, tmp_z).sum()
        
        # Yes
        if self.Vsed[jfd] > Vol_hole:
            self.Vsed[jfd] -= Vol_hole
            self.dS[ju0:ju1+1] += tmp_z - dh_tmp_0
                
        # No, Boucle pour trouver jusqu'ou remplir le trou
        else:
            hole_h = self.z_tmp[spill]+self.dS[spill] - (self.z_tmp[jfd] + self.dS[jfd])
            th0 = floor(log10(hole_h))
            
            h_sed = self.z_tmp[jfd]+self.dS[jfd]
            
            Vol_layers = 0
            # Finding the final elevation of sediment fill h_sed
            for th in range(th0, -4, -1):

                dth = 10**th
                n_lay=1
                dh_tmp = self.FlatHole(ju0, ju1, h_sed + dth)
                Vol_layers = Volume(dh_tmp_0, dh_tmp).sum()

                while Vol_layers < self.Vsed[jfd]:
                    n_lay +=1

                    # Preparing next Vol_layers test                
                    dh_tmp = self.FlatHole(ju0, ju1, h_sed + n_lay*dth)
                    Vol_layers = Volume(dh_tmp_0, dh_tmp).sum()
                else:
                    h_sed += (n_lay-1)*dth
                
            tmp_z = self.FlatHole(ju0, ju1, h_sed)
            self.Vsed[jfd] -= Volume(self.z[ju0:ju1+1]+self.dS[ju0:ju1+1], tmp_z).sum()    
            # Fill the hole up to hsed
            self.dS[ju0:ju1+1] += tmp_z - dh_tmp_0
            
            spill =0
            
        return spill

    #*********************************************************************
    
    def FlatHole(self, j0, j1, elev): #(self, dh_tmp_0, elev):
        
        #dh_tmp_0[dh_tmp_0< elev] = elev
        dh_tmp = self.z_tmp[j0:j1+1] + self.dS[j0:j1+1]
        dh_tmp[dh_tmp< elev] = elev
        
        return dh_tmp
    
    #*********************************************************************

    def FindHoles(self, slop):

        """ Finds local minima in a given profile """    

        jmins=[]

        # Finding local minima
        j=0
            
        # Find first non-flat slope
        if slop[j] == 0:
            while slop[j]==0:
                j+=1
                
        # Checking all along sent profile
        while j<len(slop):

            # If there is a hole, let's dive into it
            if slop[j]<0:

                while j<len(slop):
                    if slop[j]<=0:
                        j+=1
                    # Bottom hole is reached
                    else:
                        jmins.append(j)
                        j+=1
                        break
            else:
                j+=1
        else:
            if slop[-1]<=0:
                jmins.append(j)

        return jmins+self.start_eros

    #*********************************************************************

    def FindPeaks(self, slop):
        """ Finds local minima in a given profile  """    
        
        j=0
            
        # Find first non-flat slope
        if slop[j]==0:
            while slop[j]==0:
                j+=1     
                
        if slop[j] < 0:
            jmaxs=[j]
        else:
            jmaxs=[]

        # Finding local maxima
        while j<len(slop):
            
            if slop[j]>0:

                while j<len(slop):
                    if slop[j]>=0:
                        j+=1
                    else:
                    # Je suis au dernier maximum
                        jmaxs.append(j)
                        j+=1
                        break
                # C'etait la derniere remontee
                else:
                    jmaxs.append(j)
                    
            else:
                j+=1
        
        else:
            if slop[-1]<=0:
                jmaxs.append(j+1)
                
        jmaxs[-1]=len(slop)+1

        return jmaxs+self.start_eros
        
#*********************************************************************
#*********************************************************************
