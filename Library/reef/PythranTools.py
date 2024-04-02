import numpy as np
from functools import reduce

#pythran export vertical_eros(int64, int64, float64, float64, float64[:], float, float64[:], float64[:], float64[:])
def vertical_eros(i, riv, Vrest, eps, dh, hwb, z, dE, test):
    for j in range(i, riv+1):

        # Is there enough wave energy left to erode the sea-bed ?
        if Vrest > eps:   
            # Is water height still lower than wave base height ? In case of lagoons
            if dh[j] <= hwb:
                # Saving temporary profile for volume
                z_slice = z[j-1:j+2]
                dE_slice = dE[j-1:j+2]
                tmp = z_slice + dE_slice
                dE[j] = - Vrest * test[j] 
                z2 = z_slice+dE_slice
                # Substracting eroded volume to the total erodible volume
                #Vrest += Volume(tmp, z_slice+dE_slice).sum()
                Vrest += ((z2[:-1]+z2[1:] - tmp[:-1] - tmp[1:])/2).sum()
                
        else:
        # No more wave energy
            Vrest = 0.
            break

    return Vrest, dE

#pythran export Volume(float64[:], float64[:])
def Volume(z1, z2):

    V=np.zeros(len(z1))
    
    if len(z1)>1:
        V = (z2[:-1]+z2[1:] - z1[:-1] - z1[1:])/2    

    return V

#pythran export fill_nodes(int64, int, float64[:], float, float64, float64[:]) -> Tuple[float64, float64[:], float64[:]]
def fill_nodes(dep, end, z_tmp, ddS, Vtot, dS): #, j_save):
    for j in range (dep+1, end+1):
        Fut_zj = np.array(z_tmp[j-1:j+2])
        Fut_zj[1] = Fut_zj[0] + ddS

        # Is there enough sediments ?
        Vol_node = Volume(z_tmp[j-1:j+2], Fut_zj).sum()
        #Vol_node = ((Fut_zj[:-1]+Fut_zj[1:] - z_tmp[:-1] - z_tmp[1:])/2).sum()
        
        if Vtot >= Vol_node: 
            dS[j] += (Fut_zj[1] - z_tmp[j])                        
            z_tmp[j] = Fut_zj[1]
            Vtot -= Vol_node
            # j_save = np.append(j_save, j)
            #j_save[j - (dep+1)] = j

        # No
        else:    
            Vtot = 0.
            return Vtot, dS, z_tmp #, j_save

    return Vtot, dS, z_tmp #, j_save

#pythran export limits(float64[:], float64[:], int, int64, int64)
def limits(Fut_z, z_tmp, len_z_plus1, start_eros, dep):
    # print("first check",len(Fut_z), Fut_z.dtype)
    # print("first check z",len(z_tmp))
    # print("dep", dep, "start_eros", start_eros)

    lim_topo = np.argmax(Fut_z < z_tmp[dep:start_eros+1])-1
    # print("I went there")
    if lim_topo <= 0:
        lim_topo = len_z_plus1
    lim_hwb = np.argmax(Fut_z >= z_tmp[start_eros])
    if (lim_hwb <= 0): 
        
        if (lim_topo == len_z_plus1):
            if lim_hwb <= 0:
                return True, Fut_z
            
        else:
            lim_hwb = len_z_plus1

    end = dep + min(lim_topo, lim_hwb)
    Fut_z = Fut_z[:end-dep+1]
    Fut_z[-1] = max(
        min(Fut_z[-1], z_tmp[start_eros]), 
        z_tmp[end]
    )    
    
    return end, Fut_z





    


