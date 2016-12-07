#include <iostream>

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

int main(int argc, char* argv[])
{
    //Get some parameters
    diffeoSearchOpts theseOptions = diffeoSearchOpts(80);

    //Change the scaling
    theseOptions.thisScalingMethod = maximal;

    //Different modifiers for 2d data
    //Goal 0: 0;0
    //Rot 0: 0
    //Goal 1: +1; +5
    //Rot 1: +10° around Goal
    //Goal 2: -2; -3
    //Rot 3: -15° around +5;+5

    vector<targetModifier> allTargMods;
    allTargMods.push_back(targetModifier());
    allTargMods.push_back(targetModifier());
    allTargMods.push_back(targetModifier());
    allTargMods[0].initWithDim(2);
    allTargMods[1].initWithDim(2);
    allTargMods[2].initWithDim(2);
    VectorXd goal1(2);
    goal1 << 1.,5.;
    allTargMods[1].setNewRotation(AngleAxisd(10./180.*PI, Vector3d::UnitZ()).toRotationMatrix().topLeftCorner(2,2), goal1);
    allTargMods[1].setNewTranslation(goal1);
    VectorXd goal2(2);
    goal2 << -2.,-3.;
    VectorXd anchor2(2);
    goal1 << 5.,5.;
    allTargMods[2].setNewRotation(AngleAxisd(-15./180.*PI, Vector3d::UnitZ()).toRotationMatrix().topLeftCorner(2,2), anchor2);
    allTargMods[2].setNewTranslation(goal2);



    //Convert char* to string
    cout << argc << endl;
    vector<string> argvString;
    for (int i=0; i<argc; ++i){
        argvString.push_back( string(argv[i]) );
        cout << string(argv[i]) << endl;
    }

    if (argc<3){
        throw runtime_error("Wrong number of args");
    }

    //Simple interface

    //diffeoMethods -s [search] targetLine storageFolder path
    if (argvString[1].compare("-s")==0){
        //Search for a diffeomorphism to match targetLine
        cout << "Searching" << endl;
        string path;
        string storage;
        string time;
        if(argc>6){
            throw runtime_error("Wrong number of arguments for search");
        }
        string target = argvString[2];
        if(argc<6){
            time = "";
        }else{
            time = argvString[5];
        }
        if(argc<5){
            //Default storage path
            storage = "../result/";
        }else{
            storage = argvString[4];
        }
        if(argc<4){
            //Default data path
            path = "../dataSet/";
        }else{
            path = argvString[3];
        }
        //Create the storage folder if non existent
        //Linux specific
        (void) system(("mkdir -p "+storage+target+"/diffeo").c_str());
        //Get the data
        MatrixXd targetMat = Leph::ReadMatrix(path+target);
        VectorXd timeVec;
        if ( time.compare("")!=0 ){
            timeVec = Leph::ReadVector(path+time);
        }else{
            timeVec = VectorXd::Ones(1);
        }

        //Prepare results
        diffeoStruct thisDiffeo;
        diffeoDetails thisDetails;
        //searchDiffeoMovement(diffeoStruct& resultDiffeo, diffeoDetails& resultDetails, const MatrixXd & targetIn, const VectorXd & timeIn = VectorXd::Ones(1), const string & resultPath="")
        searchDiffeoMovement( thisDiffeo, thisDetails, targetMat, timeVec, storage+target+"/", theseOptions );

        //Apply the translation scaling here
        thisDetails.doTransform(targetMat);
        MatrixXd sourceMat =  getSourceTraj(targetMat);

        //Apply to source
        MatrixXd tauSource = sourceMat;
        //Time
        auto startT = chrono::system_clock::now();
        forwardDiffeo(tauSource, thisDiffeo);
        auto endT = chrono::system_clock::now();
        auto duration = chrono::duration_cast<std::chrono::nanoseconds>(endT - startT).count();
        cout << "time 0 : " << duration << endl;

        VectorXd blub = tauSource.col(8);
        startT = chrono::system_clock::now();
        forwardDiffeo( blub, thisDiffeo);
        endT = chrono::system_clock::now();
        duration = chrono::duration_cast<std::chrono::nanoseconds>(endT - startT).count();
        cout << "time 1 : " << duration << endl;

        //Save results
        Leph::WriteMatrix( storage+target+"/source", sourceMat, 16, "python");
        Leph::WriteMatrix( storage+target+"/target", targetMat, 16, "python");
        Leph::WriteMatrix( storage+target+"/tau_source", tauSource, 16, "python");

        /*
        //Align and scale the demonstration
        VectorXd offset = targetMat.rightCols<1>();
        targetMat = targetMat.colwise() - offset;
        VectorXd scaling = doScaling(targetMat);
        //Get the straight line to be matched onto the demonstration
        MatrixXd sourceMat = getSourceTraj(targetMat);
        //Structure to safe the diffeomorphism
        diffeoStruct thisDiffeo;
        //Perform the actual search
        iterativeSearch(sourceMat, targetMat, thisDiffeo, parametrizedOptions);
        //Apply the diffeo and save everything
        MatrixXd tauSourceMat = sourceMat;
        forwardDiffeo(tauSourceMat, thisDiffeo);
        //Safe the result
        // Get folder
        Leph::WriteMatrix( storage+target+"/source", sourceMat, 16, "python");
        Leph::WriteMatrix( storage+target+"/target", targetMat, 16, "python");
        Leph::WriteMatrix( storage+target+"/tau_source", tauSourceMat, 16, "python");
        Leph::WriteVector( storage+target+"/diffeo/scaling.txt", scaling, 16, "cpp");
        Leph::WriteVector( storage+target+"/diffeo/offset.txt", offset, 16, "cpp");
        thisDiffeo.toFolder((storage+target+"/diffeo/"));
        */
    }else if(argvString[1].compare("-a")==0){
        //diffeoMethods -a [apply] diffeoPath source result direction
        if(argc==7){
            string diffeoPath = argvString[2];
            string source = argvString[3];
            string direction = argvString[4];
            string path = argvString[5];
            string result = argvString[6];

            if (not(direction.compare("forward")||direction.compare("reverse"))){
                throw runtime_error("Direction needs to be either foward or backward");
            }
            //Get the data
            MatrixXd sourceMat = Leph::ReadMatrix(path+source);
            //Get the diffeo
            diffeoStruct thisDiffeo;
            thisDiffeo.fromFolder(diffeoPath);
            //Apply the diffeo
            if (direction.compare("forward")==0){
                forwardDiffeo(sourceMat, thisDiffeo);
            }else{
                reverseDiffeo(sourceMat, thisDiffeo);
            }

            //Save
            (void) system(("mkdir -p "+result+source).c_str());
            Leph::WriteMatrix(result+source+"/tau_source", sourceMat, 16, "python");
        }else{
            throw runtime_error("Wrong number of arguments for apply");
        }
    }else if(argvString[1].compare("-av")==0){
        //diffeoMethods -a [apply] diffeoPath sourcePoints sourceVelocity result direction
        if(argc==8){
            string diffeoPath = argvString[2];
            string sourcePoints = argvString[3];
            string sourceVelocity = argvString[4];
            string direction = argvString[5];
            string path = argvString[6];
            string result = argvString[7];

            if (not(direction.compare("forward")||direction.compare("reverse"))){
                throw runtime_error("Direction needs to be either foward or backward");
            }
            //Get the data
            MatrixXd sourcePointsMat = Leph::ReadMatrix(path+sourcePoints);
            MatrixXd sourceVelocityMat = Leph::ReadMatrix(path+sourceVelocity);
            //Get the diffeo
            diffeoStruct thisDiffeo;
            thisDiffeo.fromFolder(diffeoPath);
            //Apply the diffeo
            if (direction.compare("forward")==0){
                forwardDiffeoVel(sourcePointsMat, sourceVelocityMat, thisDiffeo);
            }else{
                reverseDiffeoVel(sourcePointsMat, sourceVelocityMat, thisDiffeo);
            }
            //Save
            (void) system(("mkdir -p "+result).c_str());
            Leph::WriteMatrix(result+sourcePoints+"/tau_source", sourcePointsMat, 16, "python");
            Leph::WriteMatrix(result+sourcePoints+"/tau_vel", sourceVelocityMat, 16, "python");
        }else{
            throw runtime_error("Wrong number of arguments for apply velocity");
        }
    }else if(argvString[1].compare("-fs")==0){
        string diffeoPath = argvString[2];
        string initPoints = argvString[3];
        string path = argvString[4];
        string storage = argvString[5];
        cout << diffeoPath << endl << initPoints << endl << path << endl << storage << endl;

        MatrixXd initPointsMat = Leph::ReadMatrix(path+initPoints);
        VectorXd tStepsVec = Leph::ReadVector(path+initPoints+"Time");

        cout << initPointsMat.rows() << " : " << initPointsMat.cols() << endl;
        cout << tStepsVec.rows() << " : " << tStepsVec.cols() << endl;


        DiffeoMoveObj thisMovement;
        thisMovement.loadFolder(diffeoPath);
        thisMovement.doInit();
        thisMovement.setZoneAi(0,-MatrixXd::Identity(thisMovement.getDimension(),thisMovement.getDimension()));
        thisMovement.setZoneFunction(ret0());
        thisMovement.setScaleFunc(standardScale());

        MatrixXd resPos;
        MatrixXd resVel;
        MatrixXd resAcc;

        size_t dim = (size_t)initPointsMat.rows();
        for (size_t i=0; i<(size_t)initPointsMat.cols();++i){
            //If the dimension is 2, set one of the modifiers
            if (dim == 2){
                thisMovement.setTargetModifier(&allTargMods[ (i%3) ]);
            }

            thisMovement.getTraj(resPos, resVel, resAcc, VectorXd(initPointsMat.col(i)), tStepsVec);
            Leph::WriteMatrix(storage+initPoints+"/pos_"+to_string(i), resPos, 16, "python");
            Leph::WriteMatrix(storage+initPoints+"/vel_"+to_string(i), resVel, 16, "python");
            Leph::WriteMatrix(storage+initPoints+"/acc_"+to_string(i), resAcc, 16, "python");
        }
    }else if(argvString[1].compare("-gv")==0){
        string diffeoPath = argvString[2];
        string name = argvString[3];
        string path = argvString[4];
        string storage = argvString[5];
        string whichSpaceStr = argvString[6];
        spaces whichSpace;

        if (whichSpaceStr.compare("demonstration")==0){
            whichSpace = demonstration;
        }else if(whichSpaceStr.compare("control")==0){
            whichSpace = control;
        }else{
            throw runtime_error("Unknown space "+whichSpaceStr);
        }

        MatrixXd pts = Leph::ReadMatrix(path+name);

        DiffeoMoveObj thisMovement;
        thisMovement.loadFolder(diffeoPath);
        thisMovement.doInit();
        if (pts.rows()==2){
            Matrix2d eV = Matrix2d::Zero();
            eV(0,0)=1.;
            eV(1,1)=2.;
            thisMovement.setZoneAi(0, -eV);
            eV(0,0)=1.;
            eV(1,1)=.05;
            thisMovement.setZoneAi(1, -eV);
            eV(0,0)=1.;
            eV(1,1)=1.;
            thisMovement.setZoneAi(2, -eV);
            VectorXi zoneNmbr(3);
            VectorXd zoneDiam(3);
            zoneNmbr << 0,1,2;
            zoneDiam << 0.8,2.25,2.75;
            sphericalZones thisZone = sphericalZones(zoneDiam, zoneNmbr);
            thisMovement.setZoneFunction(thisZone);
        }else{
            cout << "DIM is "<< thisMovement.getDimension() << endl;
            thisMovement.setZoneAi(0,-MatrixXd::Identity(thisMovement.getDimension(),thisMovement.getDimension()));
            thisMovement.setZoneFunction(ret0());
        }
        thisMovement.setScaleFunc(standardScale());

        MatrixXd vel = thisMovement.getVelocity(pts, whichSpace);
        Leph::WriteMatrix(storage+name+"/resVel", vel, 16, "python");

    }else{
        throw runtime_error("Unknwon command");
    }
    return 0;
}

