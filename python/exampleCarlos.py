from diffeoPlotUtils import *

#Get data
target = TXT2Matrix("../dataSet/carlos", fileType="python").T

dim, nPoints = target.shape
    
#Define
name='thisTarget'
path="../tmp/"
storage="../tmp/results/"
time = np.linspace(1.,0.,)

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


#Another plot with velocities
data2Vel = 200.*np.hstack((np.diff(data2,1,1), np.zeros((dim,1))));
data2Trans, data2VelTrans = transform(data2, diffeoPath=storage+name+"/", velocity=data2Vel, direction="forward", name="test", path=path, storage=storage)

#Simulate the behaviour of some neighbooring initial points
initPoints = .1*(np.random.rand(dim, 10)-0.5)+target[:,[0]];
    
tVec = np.linspace(0.0, time[0]*1.25, 1000);
posSim, velSim, accSim = simulate(diffeoPath=storage+name+"/", initPoints=initPoints, tVec=tVec, name="tmpSim", path=path, storage=storage)

plt.figure()
plt.title("Demonstration and replayed trajectories")
cList = ['g', 'b', 'r']
for k in range(initPoints.shape[1]):
    plt.plot(posSim[k][0,:], posSim[k][1,:], cList[k%3])
plt.plot(target[0,:], target[1,:], 'k', linewidth=2.)


for k in range(dim):
    plt.figure()
    plt.title("Demonstration and replayed trajectories {0}".format(k))
    for i in range(initPoints.shape[1]):
        plt.plot(tVec, posSim[i][k,:], 'g')
    plt.plot(time[0]-time, target[k,:], 'r', linewidth=2)

#Get some velocities
targetVel = np.hstack(( np.diff(target, 1,1), np.zeros((dim,1)) ))/(1./1000.)
plt.figure()
plt.title("demonstration velocities vs velocities obtained by using the diffeo")
for k in range(dim):
    for i in range(initPoints.shape[1]):
        plt.plot(velSim[i][k,:], 'g')
    plt.plot(targetVel[k,:], 'r')

plt.figure()
for k in range(0,data2.shape[1], 50):
    plt.plot( [data2[0,k], data2[0,k]+data2Vel[0,k]], [data2[1,k], data2[1,k]+data2Vel[1,k]], 'r')
for k in range(0,data2.shape[1],50):
    plt.plot( [data2Trans[0,k], data2Trans[0,k]+data2VelTrans[0,k]], [data2Trans[1,k], data2Trans[1,k]+data2VelTrans[1,k]], 'k')

for i in range(target.shape[0]):
    for j in range(i+1, target.shape[0]):
        plt.figure();
        plt.title("{0}-{1}".format(i,j))
        plt.plot(targetScaled[i,:], targetScaled[j,:], 'r') #Scaled target
        plt.plot(tauSource[i,:], tauSource[j,:], 'b') #Straightline transformed by the diffeo stored in "storage"/diffeo
        

plt.show()