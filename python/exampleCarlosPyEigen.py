import numpy as np
from diffeoPy import *
import time 

import xml.etree.ElementTree as ET

#Get data; Data as you gave me; I use only "x" and "y" (this first two dimension)
target = TXT2Matrix("../dataSet/carlos", fileType="python").T
scalingVec = TXT2Vector("../dataSet/carlosScaling", fileType="python")
dim, nPoints = target.shape
timeVec = np.linspace(1., 0., nPoints, endpoint=True)
print(target[:,0])

#Get the path options
parsXML = ET.parse('../parameters/diffeoPars.xml')
root = parsXML.getroot()

inputPath = root.find('generalOpts').find('inputPath').get('value')
resultPath = root.find('generalOpts').find('resultPath').get('value')
targetName = root.find('generalOpts').find('targetName').get('value')
scalingName = root.find('searchOpts').find('distanceScaling').get('value')
timeName = root.find('generalOpts').find('targetTime').get('value')

assert not( (inputPath is None) or (resultPath is None) or (targetName is None) or (timeName is None) ), "At least one path definition not specified or accessible in xml"

#Create the folder if non existing 
subprocess.call(["mkdir", "-p", inputPath])
subprocess.call(["mkdir", "-p", resultPath])

#Save the data in cpp format at the specified place
#And save the time file
#def Array2TXT(fileName, aArray, fileType="cpp", format="%.18e")
Array2TXT(inputPath+targetName, target)
Array2TXT(inputPath+timeName, timeVec)
if not scalingName == 'manual':
    Array2TXT(inputPath+scalingName, scalingVec)

#Get the diffeo move obj
thisMovement = PyDiffeoMoveObj()

xCpp = TXT2Matrix(resultPath+"sourceTransform")


#Get stuff to perform integration
#The demonstration takes 1sec to complete we will integrate for 1.5sec
tFinal = 1.1
deltaT = 1e-3

allT = np.arange(0., tFinal, deltaT)
nPoint = allT.size
allX = np.zeros((dim, nPoint), order='fortran')
#Create an offset that will occur at about 0.5 sec
xOffset = np.array([-0.01, 0.01])
nChange = np.floor(nPoint/2)

#Initialize the point
xCurr = np.asfortranarray(target[:,0], dtype=np.float64).copy()
print(target[:,0])
print(xCurr)
#Dummy var for velocity
vCurr = np.zeros(dim, dtype=np.float64, order='fortran')

pFig, pAx = plt.subplots(1,1)

T = time.time()
for k in range(nPoint):
    allX[:,k]=xCurr
    pAx.plot(xCurr[0], xCurr[1], 'x')
    #getVelocity(self, np.ndarray[np.float64_t, ndim=1, mode = 'c'] xIn, np.ndarray[np.float64_t, ndim=1, mode = 'c'] vOut=np.zeros(0), whichSpace = 0):
    thisMovement.getVelocity(xCurr, vCurr) #Important for you: Returns the desired velocity given a point (in the defined space)
    xCurr += vCurr*deltaT #explicit forward euler
    if k == nChange:
        thisMovement.setNewTranslation(xOffset)
T = time.time()-T
print("It took {0} seconds to perform simulations".format(T))

pAx.plot( target[0,:], target[1,:], '.-k' )
pAx.plot( target[0,:]+xOffset[0], target[1,:]+xOffset[1], '--k' )
pAx.plot( xCpp[0,:], xCpp[1,:], '.-b' )
pAx.plot( allX[0,:], allX[1,:], '-g' )#The actually followed traj
plt.show()

























