from diffeoPlotUtils import *

#Get data
target = TXT2Matrix("../dataSet/carlos", fileType="python").T

dim, nPoints = target.shape
    
#Define
name='thisTarget'
path="../tmp/"
storage="../tmp/results/"
time = np.linspace(1.,0.,nPoints)

#Get diffeo and transform
tauSource = findDiffeoAndReturnTransformed(target, name=name, path=path, storage=storage, time=time)
#Load the shifted and scaled target
targetScaled = TXT2Matrix(storage+name+"/target", fileType="python")

#Transform some other stuff
sourceScaled = TXT2Matrix(storage+name+"/source", fileType="python")
data2 = sourceScaled# + np.array([[0.1],[-0.1]]);
data2Trans = transform(data2, diffeoPath=storage+name+"/", direction="forward", name="test", path=path, storage=storage)

fig=plt.figure()
plt.plot(data2[0,:], data2[1,:], '--r')
if dim == 2:
    plot2DGrid(storage+name+"/", direction="forward", lims=None, ax=plt.gca(), N=5, N2=1000)
plt.plot(tauSource[0,:], tauSource[1,:], 'b') #Straightline transformed by the diffeo stored in "storage"/diffeo
plt.plot(targetScaled[0,:], targetScaled[1,:], 'r') #Scaled target
plt.plot(data2Trans[0,:], data2Trans[1,:], '--b')

#Streamplot
if dim==2:
    plt.figure()
    plt.plot(target[0,:], target[1,:], 'k', linewidth=2.)
    plot2DStreamlines(storage+name+"/", whichSpace="demonstration", lims=None, ax=plt.gca(), nx=300, ny=300, name= "tmpVel", path=path, storage=storage)
    #Streamplot
    plt.figure()
    plt.plot(data2[0,:], data2[1,:], 'k', linewidth=2.)
    plot2DStreamlines(storage+name+"/", whichSpace="control", lims=None, ax=plt.gca(), nx=300, ny=200, name= "tmpVel", path=path, storage=storage)


plt.show()