import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as mcoll

from mpl_toolkits.mplot3d import Axes3D

import subprocess
import re

import string

import os
cwd = os.getcwd()

#Define diffeMoethods excetubale
exePath = "./bin/Release/diffeoMethods "

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


##############################################################
##############################################################
#Simple "interace" to C++
def findDiffeoAndReturnTransformed(target, name='thisTarget', path="../tmp/", storage="../tmp/results/", time=None):
    #Make sure path exists
    subprocess.call(["mkdir", "-p", path])
    subprocess.call(["mkdir", "-p", storage+name])
    
    #Write the temporary file
    Array2TXT(path+name, target, "cpp", format="%.32e");
    timeStr = ""
    if not(time is None):
        assert time.size == target.shape[1], "Time must have the same length as target columns"
        time.resize(time.size,)
        Array2TXT(path+name+"Time", time, fileType="cpp", format="%.32e")
        timeStr = name+"Time"
    #Launch the Cpp
    
    proc = subprocess.Popen( exePath+"-s {0} {1} {2} {3}".format(name, path, storage, timeStr), shell=True )
    pp = proc.wait()
    assert pp==0, "Childprocess failed miserably with {0}".format(pp)
    
    
    #Get the result
    tauSource = TXT2Matrix(storage+name+"/tau_source", "python")
    return tauSource
##############################################################    
def transform(points, diffeoPath, velocity=None, direction="forward", name="thisTarget", path="../tmp/", storage="../tmp/results/", velName = None):
    
    velExists = not velocity==None;
    if velExists:
        assert np.all( points.shape == velocity.shape )
        if (velName==None):
            velName = name+"Vel"
    assert direction in ("forward", "reverse"), "direction needs to be forward or reverse"
    
    subprocess.call(["mkdir", "-p", path])
    subprocess.call(["mkdir", "-p", storage])
        
    #Write the temporary file
    Array2TXT(path+name, points, "cpp", format="%.32e");
    if velExists:
        Array2TXT(path+velName, velocity, "cpp", format="%.32e");
    
    #Launch the Cpp
    if not velExists:
        proc = subprocess.Popen( exePath+"-a {0} {1} {2} {3} {4}".format(diffeoPath, name, direction, path, storage), shell=True )
        pp = proc.wait()
        assert pp==0, "Childprocess failed miserably with {0}".format(pp)
        #Retrieve results
        return TXT2Matrix(storage+name+"/tau_source", fileType="python")
    else:
        proc = subprocess.Popen( exePath+"-av {0} {1} {2} {3} {4} {5}".format(diffeoPath, name, velName, direction, path, storage), shell=True )
        pp = proc.wait()
        assert pp==0, "Childprocess failed miserably with {0}".format(pp)
        #Retrieve results
        return TXT2Matrix(storage+name+"/tau_source", fileType="python"), TXT2Matrix(storage+name+"/tau_vel", fileType="python")
##############################################################     
def plot2DGrid(diffeoPath, direction = "forward", lims=None, ax=None, N=20, N2=2000, name="thisTmp", path="../tmp/", storage="../tmp/results/", dims = [0,1], allDims=None):
        #Lims is xmin xmax ymin ymax
        
        if (lims is None) and (ax is None):
            print("Define either lims or fig")
            return False
        assert direction in ("forward", "reverse"), "direction needs to be forward or reverse"
        
        if (lims is None):
            lims = np.hstack((ax.get_xlim(), ax.get_ylim()))
        
        if (ax is None):
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        allPoints = np.zeros((2,0))
        
        for k in range(N):
            #add a horizontal and a vertical line
            allPoints = np.hstack(( allPoints,  np.vstack(( (lims[0]+(lims[1]-lims[0])*(k)/(N-1))*np.ones(N2,), np.linspace(lims[2], lims[3], N2) )), np.vstack(( np.linspace(lims[0], lims[1], N2), (lims[2]+(lims[3]-lims[2])*(k)/(N-1))*np.ones(N2,) )) ))
        
        if not(allDims is None):
            #The system has more than two dimensions; All other dimensions than dims will be filled with constant terms stored in allDims
            allPointsAllDims = np.tile(allDims, [1,allPoints.shape[1]])
            allPointsAllDims[dims[0],:] = allPoints[0,:]
            allPointsAllDims[dims[1],:] = allPoints[1,:]
            allPoints = allPointsAllDims
            
        #Transform
        allPointsTau = transform(allPoints, diffeoPath=diffeoPath, velocity=None, direction=direction, name=name, path=path, storage=storage)
        
        #Plot
        for k in range(N):
            ax.plot( allPointsTau[0, 2*N2*k:2*N2*k+N2], allPointsTau[1, 2*N2*k:2*N2*k+N2], 'r' )
            ax.plot( allPointsTau[0, 2*N2*k+N2:2*N2*k+2*N2], allPointsTau[1, 2*N2*k+N2:2*N2*k+2*N2], 'k' )
        
        return ax;
##################################################
def plot2DStreamlines(diffeoPath, whichSpace="demonstration", lims=None, ax=None, nx=300, ny=300, name="thisTmp", path="../tmp/", storage="../tmp/results/", dims = [0,1], allDims=None):
    #Lims is xmin xmax ymin ymax
    if (lims is None):
        lims = np.hstack((ax.get_xlim(), ax.get_ylim()))
    
    if (ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    x,y = np.meshgrid(np.linspace(lims[0], lims[1], nx), np.linspace(lims[2], lims[3], ny))
    x.resize(x.size,)
    y.resize(y.size,)
    if allDims is None:
        X = np.vstack((x, y))
    else:
        X = np.tile(allDims, [1,x.size])
        X[allDims[0],:] = x
        X[allDims[1],:] = y
    
    subprocess.call(["mkdir", "-p", path])
    subprocess.call(["mkdir", "-p", storage+name])
    
    Array2TXT(path+name, X, fileType="cpp")
    
    proc = subprocess.Popen( exePath+"-gv {0} {1} {2} {3} {4}".format(diffeoPath, name, path, storage, whichSpace), shell=True )
    proc.wait()
    
    #Retrieve results and plot
    vel = TXT2Matrix(storage+name+"/resVel", fileType="python")
    speed = np.sqrt(np.square(vel[dims[0]])+np.square(vel[dims[1]]))
    
    ax.streamplot(x.reshape((ny,nx)), y.reshape((ny,nx)), vel[dims[0]].reshape((ny,nx)), vel[dims[1]].reshape((ny,nx)), linewidth=2*(speed.reshape((ny,nx)))/speed.max())
    
    return ax, vel
##################################################
def simulate(diffeoPath, initPoints, tVec, name="thisTmp", path="../tmp/", storage="../tmp/results/"):
    
    initPoints = initPoints.reshape((initPoints.shape[0],-1))
    tVec = tVec.reshape(tVec.size,)
    
    subprocess.call(["mkdir", "-p", path])
    subprocess.call(["mkdir", "-p", storage+name])
    
    
    Array2TXT(path+name, initPoints, fileType="cpp")
    Array2TXT(path+name+"Time", tVec, fileType="cpp")
    print("a")
    print(exePath+"-fs {0} {1} {2} {3}".format(diffeoPath, name, path, storage))
    print("b")
    proc = subprocess.Popen( exePath+"-fs {0} {1} {2} {3}".format(diffeoPath, name, path, storage), shell=True )
    pp = proc.wait()
    assert pp==0, "Childprocess failed miserably with {0}".format(pp)
    
    pos = []
    vel = []
    acc = []
    
    for k in range(initPoints.shape[1]):
        pos.append(TXT2Matrix(storage+name+"/pos_{0:d}".format(k), fileType="python"))
        vel.append(TXT2Matrix(storage+name+"/vel_{0:d}".format(k), fileType="python"))
        acc.append(TXT2Matrix(storage+name+"/acc_{0:d}".format(k), fileType="python"))
    
    return pos, vel, acc
        
    
    
    
    
    

                

          
    
    
    
    


if __name__ == "__name__":
    print("This is a module for diffeo movement plotting")
