import numpy as np
from io import StringIO
from os import path
import numba as nb


#********************************************************************************************
#********************************************************************************************


def nlines(name):
    """
    Returns the number of lines in a file
    
    Parameters:
    -----------
    
    name: str
        (Path and) Filename to read
        
    Returns:
    --------
    
    int ?: Number of lines in the file
    """
    
    f = open(name, 'r')
    text=f.readlines()
    return len(text)

#********************************************************************************************

def readfile(name):
    """ Reads values of the first two columns of a file
    
    Parameters:
    -----------

    name : str
        (Path and) Filename to read
        
    Returns:
    --------
    
    col1, col2: 1d-arrays
        Each array contains the values of a column
    """
    
    sl=open(name,"r")
    tmp=sl.read()
    col1=np.genfromtxt(StringIO(tmp), dtype='float', usecols=(0))
    col2=np.genfromtxt(StringIO(tmp), dtype='float', usecols=(1))
    sl.close()
    
    return col1, col2



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

@nb.njit#(fastmath=True)
def apply_mask_nb(z_tmp, z_tmp_init, dS):
    for i in range(len(dS)):
        if dS[i] != 0:
            z_tmp[i] = z_tmp_init[i] + dS[i]
    return z_tmp


# def FindKey(v, dic): 
#     """
#     Finds the key for a given value in a directory
    
#     Parameters:
#     -----------
    
#     v : datatype as in the values of the dictionary...
#         Value to look for
#     dic : Dictionary
#         Dictionary in which to look for
        
#     Returns:
#     --------
    
#     k: datatype as in the keys of the dictionary
#         Key corresponding to the input value
#     """
    
#     for k, val in dic.items(): 
#         if v == val: 
#             return k 


