#include <tinyxml.h>
#include <iostream>
#include <unistd.h>

#include "diffeoSearch.hpp"
#include "diffeoMovements.hpp"
#include "FileVector.h"

#include <chrono>

//#include "parameterisations.hpp"

#include<vector>

using namespace std;
using namespace Eigen;
using namespace DiffeoMethods;
using namespace DiffeoMovements;

const string xmlFile = "../parameters/diffeoPars.xml";


DiffeoMoveObj* saveDiffeoToFolder(){
    //Creates the diffeo defined in the xml
    cout << "starting search" << endl;
    //Get the path information
    double eigValOrtho;
    double breakTime;
    string inputPath, resultPath, targetName, targetTime;
    char buff[FILENAME_MAX];
    getcwd( buff, FILENAME_MAX );
    //Get the xml
    TiXmlDocument doc(xmlFile);
    if(!doc.LoadFile()){
        cerr << "Error while opening file " << xmlFile << " from " << buff << endl;
        cerr << "error #" << doc.ErrorId() << " : " << doc.ErrorDesc() << endl;
        throw;
    }
    TiXmlHandle hDoc(&doc);
    TiXmlHandle xmlOpts = hDoc.FirstChildElement("diffeo").FirstChildElement("dynamics");
    xmlOpts.FirstChildElement("eigenValOrtho").Element()->QueryDoubleAttribute("value", &eigValOrtho);
    xmlOpts.FirstChildElement("breakTime").Element()->QueryDoubleAttribute("value", &breakTime);

    xmlOpts = hDoc.FirstChildElement("diffeo").FirstChildElement("generalOpts");

    xmlOpts.FirstChildElement("inputPath").Element()->QueryStringAttribute("value", &inputPath);
    xmlOpts.FirstChildElement("resultPath").Element()->QueryStringAttribute("value", &resultPath);
    xmlOpts.FirstChildElement("targetName").Element()->QueryStringAttribute("value", &targetName);
    xmlOpts.FirstChildElement("targetTime").Element()->QueryStringAttribute("value", &targetTime);




    //Load search optionsř
    diffeoSearchOpts theseOptions = diffeoSearchOpts(xmlFile);
    cout << "searching " << theseOptions.numIterations << "diffeomorphic transformations" << endl;

    //Do the actual search
    //(const string & inputPath, const string & resultPath="", const string & targetName = "thisTarget", const string & timeName="thisTime", diffeoSearchOpts & opts= aDiffeoSearchOpt){
    /*DiffeoMoveObj thisMove = searchDiffeoMovement(inputPath, resultPath, targetName, targetTime, theseOptions);
    thisMove.doInit();

    thisMove.toFolder(resultPath);

    //Get modifier
    targetModifier * thisModifier = new targetModifier;
    thisModifier->initWithDim(thisMove.getDimension());

    thisMove.setTargetModifier(thisModifier);
    */
    MatrixXd targetIn = Leph::ReadMatrix(inputPath+targetName);
    VectorXd timeIn = Leph::ReadVector(inputPath+targetTime);

    DiffeoMoveObj * thisMove = new DiffeoMoveObj();
    diffeoStruct * thisDiffeo = new diffeoStruct();
    diffeoDetails * thisDetails = new diffeoDetails();
    searchDiffeoMovement(*thisDiffeo, *thisDetails, targetIn, timeIn, resultPath+targetName+"/", theseOptions);
    thisMove->setDiffeoStruct(*thisDiffeo);
    thisMove->setDiffeoDetailsStruct(*thisDetails);
    thisMove->doInit();

    targetModifier * thisModifier = new targetModifier;
    thisModifier->initWithDim(thisMove->getDimension());

    thisMove->setTargetModifier(thisModifier);

    thisMove->setBreakTime(breakTime);

    //This function currently only supports a single control space dynamics
    MatrixXd A0 = MatrixXd::Identity(thisMove->getDimension(), thisMove->getDimension())*(eigValOrtho);
    A0(0,0) = -1.;
    thisMove->setZoneAi(0,A0);
    thisMove->setZoneFunction(ret0()); //Single dynamics
    thisMove->setScaleFunc(standardScale()); //This should be fine in most cases

    //Save the forward transformation of a finely scaled source in order to judge matching quality
    //Use the python interfaced function to do so
    double dT = 1e-3;
    const unsigned int nSteps = ceil(timeIn(0)/dT);
    dT = timeIn(0)/((double) nSteps);
    const double iDist = (double) (targetIn.col(0)-targetIn.rightCols(1)).norm();
    VectorXd timeVec;
    timeVec.setLinSpaced(nSteps, 0., 1.5*timeIn(0));
    MatrixXd allX = MatrixXd::Zero(thisMove->getDimension(), nSteps);
    allX.col(0) = targetIn.col(0);
    cout << "initial distance is " << iDist << "; demonstration duration is " << timeIn(0) << endl;

    VectorXd vCurr(thisMove->getDimension());
    VectorXd xCurr(thisMove->getDimension());

    for (unsigned int thisStep = 1; thisStep < nSteps; ++thisStep){
        xCurr = allX.col(thisStep-1);
        vCurr = thisMove->getVelocity(xCurr);
        allX.col(thisStep) = allX.col(thisStep-1) + dT*vCurr;
    }


    Leph::WriteMatrix(resultPath+"sourceTransform", allX);
    Leph::WriteVector(resultPath+"sourceTime", timeVec);

    return thisMove;
}
