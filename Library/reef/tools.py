import numpy as np
from io import StringIO
from os import path


#********************************************************************************************
#********************************************************************************************

def Volume(z1, z2):
    """
    Measures the area between 2 profiles
    First and last points have the same elevation
    
    Parameters:
    -----------
    
    z1, z2: 1d-array
        Topographic profiles: z2 younger than z1
    
    Returns:
    --------
    
    V: 1d-array
        Array of volume along the given profile
    """

    V=np.zeros(len(z1))

    if len(z1)>1:

#         for j in range(0, len(z1)-1):
#             V[j] = (z2[j]-z1[j]+z2[j+1]-z1[j+1])*dx/2
        V = (z2[:-1]+z2[1:] - z1[:-1] - z1[1:])/2
        
    return V

#*********************************************************************

def FindKey(v, dic): 
    """
    Finds the key for a given value in a directory
    
    Parameters:
    -----------
    
    v : datatype as in the values of the dictionary...
        Value to look for
    dic : Dictionary
        Dictionary in which to look for
        
    Returns:
    --------
    
    k: datatype as in the keys of the dictionary
        Key corresponding to the input value
    """
    
    for k, val in dic.items(): 
        if v == val: 
            return k 
        
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

def CheckFile(name):
    """ 
    Checks if a file exists
    
    Parameters:
    -----------
    
    name : str
     (Path and) filename to check
     
    Returns:
    --------
    
    boolean
    """
    
    return path
