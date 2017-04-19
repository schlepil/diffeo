# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libcpp.string cimport string

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as mcoll

from mpl_toolkits.mplot3d import Axes3D

import subprocess
import os
cwd = os.getcwd()

#Some text input output
##############################################################
def TXT2Matrix(fileName, fileType="cpp"):
    aArray = np.loadtxt(fileName)
    if fileType == "python":
        return aArray
    elif fileType == "cpp":
        dim = int(aArray[0])
        nPoints = int(aArray[1])
        outArray = np.zeros((dim, nPoints))
        aArray = aArray[2::]
        for k in range(nPoints):
            outArray[:,k] = aArray[dim*k:dim*(k+1)]
        return outArray
    else:
        return False
##############################################################
def TXT2Vector(fileName, fileType="cpp"):
    aArray = np.loadtxt(fileName)
    if fileType == "python":
        return aArray
    elif fileType == "cpp":
        return np.resize( aArray[1::], aArray[0] )
    else:
        return False
##############################################################
def Array2TXT(fileName, aArray, fileType="cpp", format="%.18e"):
    if fileType == "cpp":
        np.savetxt(fileName, np.hstack(( aArray.shape, aArray.T.reshape(aArray.size,) )), fmt=format)    
    elif fileType == "python":
        np.savetxt(fileName, aArray, fmt=format)
    else:
        return False
    return True
############################################################## 


# Get a interface of the class
# Only the most important function are exposed

cdef extern from "diffeoMovements.hpp":
    cdef cppclass DiffeoMoveObj:
        DiffeoMoveObj() except + #Construction
        void doInit() #do some internal initialisation
        void loadFolder(const string & aFolder) #Load the diffeo stored in a folder
        void toFolder(const string & aFolder) #and the corresponding details (Mainly for dynamics and some numerical hacks)
        void getVelocityPy(double * ptPtr, double *velPtr, unsigned long nPt, unsigned int thisSpaceAsInt) #Important for you: Returns the desired velocity given a point (in the defined space)
        int setNewTranslationPy(double * newTranslationPtr) #Moving target
        void setBreakTime( const double & newBreakTime ) #Sets the "size" of the sphere around the target when to start decelerating
        const int getDimension() #Return the dimension of the dynamical system

cdef extern from "diffeoPy.hpp":
    DiffeoMoveObj* saveDiffeoToFolder()
    

cdef class PyDiffeoMoveObj:
    #Python interface
    #To initialize, the function in diffeoPy will be called in order to get a 
    #full fledged movement
    cdef DiffeoMoveObj * cDM
    def __cinit__(self):
        self.cDM = saveDiffeoToFolder()
    def __dealloc__(self):
        del self.cDM
    
    def doInit(self):
        self.cDM.doInit() #From the doc: . can be used on objects and pointers to objects
        return 0;
    def getDimension(self):
        return self.cDM.getDimension()
    def loadFolder(self, str folderPath):
        #I have no clue if that is the best way but it seems to work
        self.cDM.loadFolder(<string> bytes(folderPath, 'utf-8'))
    def getVelocity(self, np.ndarray[np.float64_t, mode = 'fortran'] xIn, np.ndarray[np.float64_t, mode = 'fortran'] vOut=np.zeros((0,), dtype=np.float64, order='F'), whichSpace = 0):
        #ATTENTION this function is designed for double precision vectors of the "right" size; No size check-up is performed
        #It will stock the results in the vOut vector if provided; if not provided than a new one
        #will be created (with the overhead associated)
        #The gain performance, correct parameter types are demanded instead of being ensured
        print("xPy:\n{0}".format(xIn))
        assert (whichSpace in [0,1]), "which space has to be 0 or 1"
        whichSpaceC = <unsigned int> whichSpace
        #If no input is given
        if vOut.size==0:
            vOut = np.zeros_like(xIn)
            
        #Get pointers and call actual c function
        cdef double * xPtr = <double *> xIn.data
        cdef double * vPtr = <double *> vOut.data
        self.cDM.getVelocityPy(xPtr, vPtr, <unsigned long> xIn.size//self.cDM.getDimension(), whichSpaceC)
        return vOut
    
    def setNewTranslation(self, np.ndarray[np.float64_t, ndim=1, mode = 'c'] newOffset ):
        cdef double * oPtr = <double *> newOffset.data
        return self.cDM.setNewTranslationPy(oPtr)
    
    def setBreakTime(self, newBreakTime):
        self.cDM.setBreakTime(<double> newBreakTime)
        return 0
    
        
        
        
        