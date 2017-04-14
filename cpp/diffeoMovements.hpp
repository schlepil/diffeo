#ifndef DIFFEO_MOVEMENTS_HPP
#define DIFFEO_MOVEMENTS_HPP

#include <math.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <vector>
#include <functional>

#include "diffeoMethods.hpp"
#include "diffeoSearch.hpp"
#include "thingsThatShouldExist.hpp"

using namespace std;
using namespace Eigen;
using namespace DiffeoMethods;
using namespace thingsThatShouldExist;

namespace DiffeoMovements{
    //////////////////////////////////////////////////////////////////////////////
    enum spaces {demonstration, control};
    //////////////////////////////////////////////////////////////////////////////
    //The main class and its helpers
    //A funtor to determine the zone
    //Return 0
    //#TBD find a template free way to accept Matrices and Vectors and return either int or vector of int for zone functions
    struct ret0{
        inline int operator() (const VectorXd & ptX) const { return 0; };
    };

    class sphericalZones{
        private:
            VectorXd _diametersSquared;
            VectorXd _diameters;
            VectorXi _zoneNumbers;
            size_t _size;

        public:
            sphericalZones(){
                _diameters = VectorXd::Zero(1);
                _diametersSquared = VectorXd::Zero(1);
                _zoneNumbers = VectorXi::Zero(1);
                _size = 0;
            }
            //Initialize with new diameters and zone numbers
            sphericalZones(const VectorXd & newDiameters, const VectorXi & newNumbers){
                if (not(newDiameters.size()==newNumbers.size())){
                    throw runtime_error("There need to be as many zones as zone numbers");
                }
                _diameters = newDiameters;
                VectorXi idx = doSort(_diameters);
                _diametersSquared = _diameters.array().square();
                _zoneNumbers = VectorXi::Zero(newDiameters.size());
                for (size_t i=0; i<(size_t)newDiameters.size(); ++i){
                    _zoneNumbers(i) = newNumbers(idx(i));//zonenumber
                }
                _size = (size_t) newDiameters.size();
            }
            //set new diameters and zone numbers
            void setNewDiametersAndNumbers(VectorXd newDiameters, VectorXi newNumbers){
                if (not(newDiameters.size()==newNumbers.size())){
                    throw runtime_error("There need to be as many zones as zone numbers");
                }
                _diameters = newDiameters;
                VectorXi idx = doSort(_diameters);
                _diametersSquared = _diameters.array().square();
                for (size_t i=0; i< (size_t) newDiameters.size(); ++i){
                    _zoneNumbers(i) = newNumbers(idx(i));
                }
                _size = (size_t) newDiameters.size();
            }

        //The zones are concentric circles (in which ever space considered). The current zone is the smallest circle containing the point
        //diameter is in ascending order
        inline int operator() (const VectorXd & pt) const {
            double thisNorm = pt.squaredNorm();
            for (size_t i=0; i<_size; ++i){
                if (thisNorm <= _diametersSquared(i)){
                    return (int) _zoneNumbers(i);
                }
            }
            return (int) _zoneNumbers(_size-1);//Default to last one
        }
    };
//////////////////////////////////////////////////////////////////////////////
    //A functor to scale the velocity
    /*
    stepSize is the the norm of the of the vector (start point - final point) divided by the number of points in the demonstration.
    If trying to obtain "exactly" the demonstration points set stepTime to (total demonstration time)/(number of demonstration points)
    If trying to integrate the continuous time equation d/dt x = f(x) (so you actually want to obtain the velocity) set stepTime to 1.
    If you want to get x' = x_0 + int_0^T (f(x)) then set stepTime to T
    #TBD: Try to template the size to enhance perfs
    */
    struct standardScale{
        template<typename VM1>
        inline void operator () (const VM1 & ptY, VM1 & ptYd, const double & controlSpaceVelocity, const double & breakTime = .15, const double & stepTime = 1.) {

            VectorXd ptYNorm = ptY.colwise().norm(); //Each column is treated as the velocity associated to a point
            ptYNorm = ptYNorm.array() + 1e-9;
            VectorXd thisVelocity = VectorXd::Constant(ptY.cols(), controlSpaceVelocity); //If a point is outside the breaking region; its norm should be equal to controlSpaceVelocity

            for (size_t i=0; i<(size_t) ptY.cols();++i){
                if( ptYNorm(i) < breakTime*controlSpaceVelocity ){
                    thisVelocity(i) = thisVelocity(i)*ptYNorm(i)/(breakTime*controlSpaceVelocity);//Scale the velocity norm linear from zero at the origin to 1 at the border of the breaking zone
                }
            }
            //Normalize the velocity with the desired velocity factor
            ptYd = stepTime * ( (ptYd.cwiseProduct( thisVelocity.replicate( ptY.rows(), 1 ) )).cwiseQuotient( ptYNorm.replicate(ptY.rows(), 1) ) );
        }
    };
//////////////////////////////////////////////////////////////////////////////
    //Helper rotation diffeo struct
    /*struct RotDiffeo{
        //This has to be made properly
        //Handle with care
        public:
            MatrixXd Rot;
            MatrixXd RotInv;
            VectorXd center;
            double beta;
            double alpha;
    }
    */
    //Quick n Dirty implementation to locally optimize mouvements
    /*
    This scale function composes a modification of the control space velocity with the actual scaling
    The control space velocity modification has the same effect as composing the actual diffeo with a
    second diffeo composed of transformations of the form
    x' = c+R(alpha*exp(-beta^2.||x-c||^2)).(x-c)
    where R(alpha*exp(-beta^2.||x-c||^2)) is a rotation matrix
    */
/*
    class modifyingScale{
        public:
            vector<RotDiffeo *> diffeoVect;
            modifyingScale(){
                diffeoVect.clear()
            }
            //Modify the velocity
            template<typename VM1>
            inline void operator () (const VM1 & ptY, VM1 & ptYd, const double & controlSpaceVelocity, const double & breakTime = .15, const double & stepTime = 1.) {
                MatrixXd R = MatrixXd::Identity(ptY.rows());
                MatrixXd dR = MatrixXd::Identity(ptY.rows());
                VectorXd thisC;
                unsigned nPt = ptY.cols();
                for(vector<RotDiffeo *>::reverse_iterator rit = diffeoVect.rbegin(); rit != diffeoVect.rend(); ++rit)
                    {
                    //Loop over transformations
                    thisC = (*rit).center;
                    MatrixXd dX = ptY.colwise()-thisC;
                    VectorXd angles = (*rit).alpha*(-((*rit).beta*(*rit).beta)*(dX.cwiseProduct(dX))).colwise().sum().array().exp();
                    VectorXd sinAngles = angles.array().sin();
                    VectorXd cosAngles = angles.array().cos();
                    //Loop over points
                    for (unsigned j = 0; j++ < nPt; ){
                        //Assemble the matrices
                        R = MatrixXd::Identity(ptY.rows());
                        dR = MatrixXd::Identity(ptY.rows());
                        R(0,0)=cosAngles(j);
                        R(1,1)=cosAngles(j);
                        R(0,1)=-sinAngles(j);
                        R(1,0)= sinAngles(j);
                        dR(0,0)=-sinAngles(j);
                        dR(1,1)=-sinAngles(j);
                        dR(0,1)=-cosAngles(j);
                        dR(1,0)= cosAngles(j);
                        R = (rit -> RotInv)*R*(rit -> Rot);
                        dR = (-2.*(rit -> beta)*(rit->beta)*angles(j)*(rit->RotInv)*dR*(rit->Rot))*(dX.col(j)*(dX.col(j).transpose()));

                        ptYd.col(j)=(R+dR)*ptYd.col(j);
                    }
                }
                //Limit velocity of transformed

            }
    }
*/
//////////////////////////////////////////////////////////////////////////////
    struct diffeoDetails{
        VectorXd _scaling;
        VectorXd _offset;
        MatrixXd _linGainRot;
        double _controlSpaceVelocity;

        MatrixXd calcNtransform( const MatrixXd & targetIn, const VectorXd & timeIn = VectorXd::Ones(1), diffeoSearchOpts & opts = aDiffeoSearchOpt){
            _offset = targetIn.rightCols<1>();
            MatrixXd target = targetIn.colwise() - targetIn.rightCols<1>();
            _scaling = doScaling(target, opts);
            //Get the other details
            //The average velocity in control space is the norm of the starting point divided by the time taken
            _controlSpaceVelocity = (target.leftCols<1>().norm())/timeIn(0);
            //The rotation is the source vector concatenated with his null space
            VectorXd origV = target.leftCols<1>();
            origV /= origV.norm();
            FullPivLU<MatrixXd> lu(origV.transpose());
            MatrixXd rotAdd = lu.kernel();
            _linGainRot = MatrixXd::Zero(origV.size(), origV.size());
            _linGainRot.topRows<1>() = origV;
            _linGainRot.bottomRows(origV.size()-1) = rotAdd.transpose();
            return target;
        }

        MatrixXd calcNtransform(const MatrixXd & targetIn, diffeoSearchOpts & opts){
            return calcNtransform(targetIn, VectorXd::Ones(1), opts);
        }

        //Perform the translation/scaling action
        template<typename VoM>
        inline void doTransform(VoM & points){
            points = (points.colwise()-_offset).cwiseQuotient( (_scaling.replicate(1,points.cols())) );
        }
        template<typename VoM1, typename VoM2>
        inline void doTransform(VoM1 & points, VoM2 & velocity){
            points = (points.colwise()-_offset).cwiseQuotient( (_scaling.replicate(1,points.cols())) );
            velocity = velocity.cwiseQuotient( (_scaling.replicate(1,velocity.cols())) );
        }
        template<typename VoM1, typename VoM2>
        inline void doTransformVel(VoM1 & velocity){
            velocity = velocity.cwiseQuotient( (_scaling.replicate(1,velocity.cols())) );
        }
        template<typename VoM1, typename VoM2, typename VoM3>
        inline void doTransform(VoM1 & points, VoM2 & velocity, VoM3 & accleration){
            points = (points.colwise()-_offset).cwiseQuotient( (_scaling.replicate(1,points.cols())) );
            velocity = velocity.cwiseQuotient( (_scaling.replicate(1,velocity.cols())) );
            accleration = velocity.cwiseQuotient( (_scaling.replicate(1,accleration.cols())) );
        }
        template<typename VoM>
        inline void undoTransform(VoM & points){
            points = (points.cwiseProduct( (_scaling.replicate(1, points.cols())) )).colwise() + _offset;
        }
        template<typename VoM1, typename VoM2>
        inline void undoTransform(VoM1 & points, VoM2 & velocity){
            points = (points.cwiseProduct( (_scaling.replicate(1, points.cols())) )).colwise() + _offset;
            velocity = velocity.cwiseProduct( (_scaling.replicate(1, velocity.cols())) );
        }
        template<typename VoM1>
        inline void undoTransformVel(VoM1 & velocity){
            velocity = velocity.cwiseProduct( (_scaling.replicate(1, velocity.cols())) );
        }
        template<typename VoM1, typename VoM2, typename VoM3>
        inline void undoTransform(VoM1 & points, VoM2 & velocity, VoM3 & acceleration){
            points = (points.cwiseProduct( (_scaling.replicate(1, points.cols())) )).colwise() + _offset;
            velocity = velocity.cwiseProduct( (_scaling.replicate(1, velocity.cols())) );
            acceleration = acceleration.cwiseProduct( (_scaling.replicate(1, acceleration.cols())) );
        }

        bool toFolder(const string & path){

            //Make the subdir but not the path
            (void) system( ("mkdir -p "+path+"/diffeo/details").c_str() );
            //Check how to catch errors here

            Leph::WriteMatrix(path+"/diffeo/details/linGainRot.txt", _linGainRot, 32, "cpp");
            Leph::WriteVector(path+"/diffeo/details/scaling.txt", _scaling, 32, "cpp");
            Leph::WriteVector(path+"/diffeo/details/offset.txt", _offset, 32, "cpp");
            Leph::WriteVector(path+"/diffeo/details/cSpaceVel.txt", VectorXd::Ones(1)*_controlSpaceVelocity, 32, "cpp");

            return true;
        }

        bool fromFolder(const string & path){
            _linGainRot = Leph::ReadMatrix(path+"/diffeo/details/linGainRot.txt");
            _scaling = Leph::ReadVector(path+"/diffeo/details/scaling.txt");
            _offset = Leph::ReadVector(path+"/diffeo/details/offset.txt");
            _controlSpaceVelocity = (double) (Leph::ReadVector(path+"/diffeo/details/cSpaceVel.txt"))(0);

            if (not(_linGainRot.cols()==_linGainRot.rows())){
                cerr << "Rotation matrix needs to be square" << endl;
                return false;
            }
            if (not((_linGainRot.rows() == _scaling.rows()) && (_linGainRot.rows() == _scaling.rows()))){
                cerr << "Dimensions do not agree" << endl;
                return false;
            }

            return true;
        }
    };
//////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////
class targetModifier{

    private:
        MatrixXd _rotation;
        MatrixXd _rotationInv;
        VectorXd _translation;
        VectorXd _anchorPoint;
        VectorXd _deltaForwardTransform;

    public:
        targetModifier(){};
        ////////////////
        void initWithDim(const int newDim){
            _rotation = MatrixXd::Identity(newDim, newDim);
            _rotationInv = MatrixXd::Identity(newDim, newDim);
            _translation = VectorXd::Zero(newDim);
            _anchorPoint = VectorXd::Zero(newDim);
            _deltaForwardTransform = VectorXd::Zero(newDim);
        }
        ////////////////////////////////////////////
        void setNewRotation(const MatrixXd & newRotation, const VectorXd & newAnchorPoint){
            if (not(newRotation.cols()==newRotation.rows())){
                throw runtime_error("Rotation matrix needs to be square");
            }
            if (not(newRotation.cols()==_anchorPoint.size())){
                throw runtime_error("Rotation matrix and anchorPoint do not have the same dimension");
            }
            _rotation = newRotation;
            _rotationInv = newRotation.inverse();
            _anchorPoint = newAnchorPoint;
            _deltaForwardTransform = _translation+_anchorPoint;
        }
        ////////////////
        template<int dim>
        void setNewRotation(const Matrix<double, dim,dim> & newRotation){
            setNewRotation(newRotation, _anchorPoint);
        }
        ////////////////
        void setNewRotation(const VectorXd & newAnchorPoint){
            setNewRotation(_rotation, newAnchorPoint);
        }
        ////////////////////////////////////////////
        void setNewTranslation(const VectorXd & newTranslation ){
            _translation = newTranslation;
            _deltaForwardTransform = _translation+_anchorPoint;
        }
        ////////////////////////////////////////////
        //The applied transformation is equal to as shift by -_translation followed by a rotation around _anchorPoint
        template<typename VoM>
        inline void doTransform(VoM & pt){
            pt = (_rotation*(pt.colwise()-_deltaForwardTransform)).colwise()+_anchorPoint;
        }
        ///////////////////
        template<typename VoM1, typename VoM2>
        inline void doTransform(VoM1 & pt, VoM2 & vel){
            pt = (_rotation*(pt.colwise()-_deltaForwardTransform)).colwise()+_anchorPoint;
            vel = _rotation*vel;
        }
        ///////////////////
        template<typename VoM1>
        inline void doTransformVel(VoM1 vel){
            vel = _rotation*vel;
        }
        ///////////////////
        template<typename VoM1, typename VoM2, typename VoM3>
        inline void doTransform(VoM1 & pt, VoM2 & vel, VoM3 & acc){
            pt = (_rotation*(pt.colwise()-_deltaForwardTransform)).colwise()+_anchorPoint;
            vel = _rotation*vel;
            acc = _rotation*acc;
        }
        ///////////////////
        template<typename VoM>
        inline void undoTransform(VoM & pt){
            pt = (_rotationInv*(pt.colwise()-_anchorPoint)).colwise()+_deltaForwardTransform;
        }
        ///////////////////
        template<typename VoM1, typename VoM2>
        inline void undoTransform(VoM1 & pt, VoM2 & vel){
            pt = (_rotationInv*(pt.colwise()-_anchorPoint)).colwise()+_deltaForwardTransform;
            vel = _rotationInv*vel;
        }
        ///////////////////
        template<typename VoM1>
        inline void undoTransformVel(VoM1 & vel){
            vel = _rotationInv*vel;
        }
        ///////////////////
        template<typename VoM1, typename VoM2, typename VoM3>
        inline void undoTransform(VoM1 & pt, VoM2 & vel, VoM3 & acc){
            pt = (_rotationInv*(pt.colwise()-_anchorPoint)).colwise()+_deltaForwardTransform;
            vel = _rotationInv*vel;
            acc = _rotationInv*acc;
        }
    };


//////////////////////////////////////////////////////////////////////////////
    class DiffeoMoveObj{
        //friend KickMove;
        private:
            //Stuff defining the control law in control space
            //The resulting velocity in control space is scale(A[_fZone(x)].x)
            vector< MatrixXd* > _Alist;
            function<int (const VectorXd & ptX)> _fZone;
            function<void (const VectorXd & ptY, VectorXd & ptYd, const double & controlSpaceVelocity, const double & breakTime, const double & stepTime)> _scaleFunc;

            //Preliminary transformations between demonstration space and scaled demonstration space
            diffeoDetails _thisDiffeoDetails;
            //The information about the actual diffeo
            diffeoStruct _thisDiffeo;
            //The target modifier is public
            //Fixed explicit euler forward integration time-step
            //TBD This is ugly, use a existing solver like gnu runge-kutta
            double _dTint;
            double _dTnumericDiff;
            double _breakTime;

            int _dim;

            bool _isInit;

            bool _isFinished;
            double _finishedNormSquare;

        public:
            const bool & isFinished;
            //A target modifier
            targetModifier * _thisModifier;

            ///////////////////////////////////////////////
            DiffeoMoveObj():isFinished(_isFinished){

                _Alist.clear();
                _fZone = ret0();
                _isInit = false;
                _dim = -1;
                _thisModifier = nullptr;
                _isFinished = false;
                _finishedNormSquare = 0.05*0.05;//#TBD do sth smarter to check  finished Hardcoded
            }
            ///////////////////////////////////////////////
            DiffeoMoveObj(int dim, double finishedNorm = 0.05):isFinished(_isFinished){

                _Alist.clear();
                _fZone = ret0();
                _isInit = false;
                _dim = dim;
                _thisModifier = nullptr;
                _isFinished = false;
                _finishedNormSquare = finishedNorm*finishedNorm;//#TBD do sth smarter to check  finished Hardcoded
            }
            //////////////////////////////////////////////
            void loadFolder(const std::string & aFolder)
            {
                _thisDiffeo.fromFolder(aFolder);
                _thisDiffeoDetails.fromFolder(aFolder);
                //Add some assertions here
                _isInit = false;
            }
            //////////////////////////////////////////////
            void toFolder(const std::string & aFolder)
            {
                _thisDiffeo.toFolder(aFolder);
                _thisDiffeoDetails.toFolder(aFolder);
            }
            //////////////////////////////////////////////
            void doInit(){

                _thisModifier = nullptr;

                _dTint = 1e-3;
                _dTnumericDiff = 1e-4;
                _breakTime = 0.15;

                //Add some asserts
                _dim = _thisDiffeo.centers.rows();

                _isInit = true;
            }
            //////////////////////////////////////////////
            const bool isInit(){
                return _isInit;
            }
            //////////////////////////////////////////////
            const int getDimension(){
                if (not _isInit){
                    return -1;
                }else{
                    return _dim;
                }
            }
            //////////////////////////////////////////////
            void setLinGainRot(MatrixXd newLinGainRot){

                MatrixXd linGainRotInv = newLinGainRot.inverse();
                _thisDiffeoDetails._linGainRot = newLinGainRot;
                for ( size_t i=0; i<_Alist.size(); i++){
                    //Undo old rotation
                    *_Alist[i] = (linGainRotInv.transpose())*(*_Alist[i])*(linGainRotInv);
                    //Apply new
                    *_Alist[i] = (_thisDiffeoDetails._linGainRot.transpose())*(*_Alist[i])*(_thisDiffeoDetails._linGainRot);
                }
            }
            //////////////////////////////////////////////
            void setControlSpaceVelocity(const double & newConstrolSpaceVelocity){
                assert(newConstrolSpaceVelocity>0. && "negative stepsize!");
                _thisDiffeoDetails._controlSpaceVelocity = newConstrolSpaceVelocity;
            }
            //////////////////////////////////////////////
            const double getConstrolSpaceVelocity(){
                return _thisDiffeoDetails._controlSpaceVelocity;
            }
            //////////////////////////////////////////////
            void setBreakTime( const double & newBreakTime ){
                assert(newBreakTime>=0. && "break time needs to be non-negative");
                _breakTime = newBreakTime;
            }
            //////////////////////////////////////////////
            const double getBreakTime(){
                return _breakTime;
            }
            //////////////////////////////////////////////
            void setDiffeoStruct(const diffeoStruct & aDiffeo){
                if(not _isInit){
                    _thisDiffeo = aDiffeo;
                    _dim = aDiffeo.centers.rows();
                }else{
                    assert(aDiffeo.centers.rows()==_dim);
                }
            }
            //////////////////////////////////////////////
            diffeoStruct & getDiffeoStuct(){
                return _thisDiffeo;
            }
            //////////////////////////////////////////////
            void setDiffeoDetailsStruct(const diffeoDetails & newDetails){
                _thisDiffeoDetails = newDetails;
                //Also add some asserts
            }
            //////////////////////////////////////////////
            diffeoDetails & getDiffeoDetailsStruct(){
                return _thisDiffeoDetails;
            }
            //////////////////////////////////////////////
            //////////////////////////////////////////////
            template<typename Mat>
            void setZoneAi( const size_t i, const Mat & Ai ){
                //assert(i<nZones && "out of range");
                //cout << _linGainRot << endl;
                while (_Alist.size()<=i){
                    MatrixXd* thisMat = new MatrixXd;
                    *thisMat = MatrixXd::Identity(Ai.rows(), Ai.cols());
                    _Alist.push_back( thisMat );
                    cout << "Addr " << thisMat << endl;
                }

                MatrixXd* thisMat = new MatrixXd;
                *thisMat = _thisDiffeoDetails._linGainRot.transpose()*Ai*_thisDiffeoDetails._linGainRot;
                _Alist[i] = thisMat;
                cout << "Addr " << thisMat << endl;
                cout << *_Alist[i] << endl;
            }
            //////////////////////////////////////////////
            void setZoneFunction( std::function<int (const VectorXd & pt)> fZone ){
                _fZone = fZone;
            }
            //////////////////////////////////////////////
            void setScaleFunc( std::function<void ( const VectorXd & ptY, VectorXd & ptYd, const double & controlSpaceVelocity, const double & breakTime, const double & stepTime )> newScaleFunc ){
                _scaleFunc = newScaleFunc;
            }
            //////////////////////////////////////////////
            void setScaling(const VectorXd & newScaling){
                _thisDiffeoDetails._scaling = newScaling;
            }
            //////////////////////////////////////////////
            void setTargetModifier(targetModifier * newTargetModifier = nullptr){
                //Attention a copy of the given targetModifier is used; Only modify the one stored in the object
                if (newTargetModifier){
                    _thisModifier = new targetModifier(*newTargetModifier);
                }else{
                    _thisModifier = new targetModifier();
                }
            }
            //////////////////////////////////////////////
            //void setNewTranslation(const VectorXd & newTranslation )
            //This is not how one wants this to be done if coding properly
            //but i refrain from exposing the modifiers to python at the moment
            int setNewTranslationPy(double * newTranslationPtr){
                if(!_thisModifier){
                    cout << "Can not set a new translation for a modifier that does not exist!" << endl;
                    return 1; //Failed to set
                }
                //"Useless" copy
                VectorXd newTranslationVec(_dim);
                for (size_t i=0; i<_dim; ++i){
                    newTranslationVec(i) = newTranslationPtr[i];//No verification is performend
                }
                _thisModifier->setNewTranslation(newTranslationVec);
                return 0;
            }
            //////////////////////////////////////////////
            //This is a convenience function for python interface; its not the most efficient but will do for the moment
            //This does 0 verification what so ever so be reasonnably careful
            void getVelocityPy(double * ptPtr, double *velPtr, unsigned int thisSpaceAsInt=0){
                //Get a single point as array pointer and the according space (0 for demonstration; 1 for control)
                //and computes the corresponding veloctiy and puts it into the designed place
                //Get the values from the pointer
                VectorXd pt(_dim);
                VectorXd vel(_dim);
                size_t i;
                spaces thisSpace = static_cast<spaces>(thisSpaceAsInt); //thisSpaceAsInt must be 0 or 1
                for (i=0; i<_dim; ++i){
                    pt(i) = ptPtr[i];
                }
                vel = getVelocity(pt, thisSpace);
                //Put it into the array
                for (i=0; i<_dim; ++i){
                    velPtr[i] = vel(i);
                }
                return;
            }

            //////////////////////////////////////////////
            template<typename MoV1>
            MatrixXd getVelocity( MoV1 & pt,  const spaces & thisSpace=demonstration){

                //Get the velocity of points in the control or demonstration space
                const size_t dim = pt.rows();
                const size_t nPt = pt.cols();
                MoV1 vel = MoV1::Zero(dim, nPt);
                VectorXd thisVel = VectorXd::Zero(dim);
                //Transform from demonstration to control
                if(thisSpace==demonstration){
                    if(_thisModifier){
                        _thisModifier->doTransform(pt);
                    }
                    _thisDiffeoDetails.doTransform(pt);
                    reverseDiffeo(pt, _thisDiffeo);
                }
                //This might turn a bit slow because zone and scale have to be caled each time; But maybe it's not that bad
                //Loop over all points
                for (size_t i=0; i<(size_t)pt.cols(); ++i){
                        //cout << _fZone(pt.col(i)) << endl;
                        thisVel = (*_Alist[_fZone(pt.col(i))])*((pt.col(i)));
                        _scaleFunc(pt.col(i), thisVel, _thisDiffeoDetails._controlSpaceVelocity, _breakTime, 1. );
                        vel.col(i) = thisVel;
                }
                //From control to demonstration
                if(thisSpace==demonstration){
                    /*
                    //Apply jacobian #TBD make sure that the stored jacobian is the actually the jacobian of the diffeo and not its inverse
                    for (size_t i=0; i<(size_t)pt.cols(); ++i){
                        vel.col(i) = theseJacs[i]*vel.col(i);
                    }
                    */
                    //Transform points and velocities back to demonstration space
                    forwardDiffeoVel(pt, vel, _thisDiffeo);
                    _thisDiffeoDetails.undoTransformVel(vel);
                    if(_thisModifier){
                        _thisModifier->undoTransformVel(vel);
                    }
                }
                return vel;
            }
            //////////////////////////////////////////////
            /**
            Most important functions
            Takes a point in the (unscaled) demonstration space (i.e. the current state of the system) ptX and
            a vector of arbitrary length containing time stamps tSteps and one to three matrices for storing the results
            The stored result corresponds to the state/velocity/acceleration of the system if following the flow imposed by the diffeo mvt.
            */

            template<typename M1, typename V1, typename V2>
            void getTraj( M1 & nextPtX, const V1 & ptX, const V2 & tSteps ){
                //Take the given point and calculate the forward trajectory
                //Attention tSteps has to be in "relative time" -> the current position corresponds to t=0
                /*assert (nextPtX.rows() == _dim && "Wrong dimension");
                assert (nextPtX.cols() == tSteps.size() && "points and timesteps incompatible");
                */
                assert (ptX.size() == _dim && "Wrong dimension");

                //Apply target modifier if existent
                VectorXd ptPrime = ptX;
                if(_thisModifier){
                    _thisModifier->doTransform(ptPrime);
                }

                //Transform it into the scaled demonstration space
                _thisDiffeoDetails.doTransform(ptPrime);


                //Tranform the point into the control space
                reverseDiffeo(ptPrime, _thisDiffeo);

                //Check if finished
                _isFinished = (ptPrime.squaredNorm() <= _finishedNormSquare);

                //Integrate (explicit fixed step forward euler) in the transformed space
                //TBD nahhh change this
                double tCurr = 0.;
                int indCurr = 0;

                nextPtX = MatrixXd::Zero(_dim, tSteps.size());

                int thisZone, thisZone2;
                MatrixXd thisA;

                double thisdT;
                VectorXd ptPrimed(_dim);
                //Initialise
                thisZone = _fZone(ptPrime);
                thisA = *_Alist[thisZone];
                while (tCurr < tSteps[tSteps.size()-1]){

                    if (tCurr+_dTint>tSteps[indCurr]){
                        //reaching a demanded timepoint
                        thisdT = tSteps[indCurr]-tCurr;
                        ptPrimed = thisA*ptPrime;
                        _scaleFunc(ptPrime, ptPrimed, _thisDiffeoDetails._controlSpaceVelocity, _breakTime, 1. );
                        //Integrate
                        ptPrime+=thisdT*ptPrimed;
                        //Save
                        //pos
                        nextPtX.col(indCurr) = ptPrime;

                        tCurr = tSteps(indCurr);
                        indCurr += 1;
                    }else{
                        //integrating
                        ptPrimed = thisA*ptPrime;
                        _scaleFunc(ptPrime, ptPrimed, _thisDiffeoDetails._controlSpaceVelocity, _breakTime, 1. );
                        //Integrate
                        ptPrime+=_dTint*ptPrimed;
                        tCurr+=_dTint;
                    }
                    thisZone2 = _fZone(ptPrime);
                    if(thisZone!=thisZone2){
                        thisZone = thisZone2;
                        thisA = *_Alist[thisZone];
                    }
                }

                //Transform the position and velocity from control to demonstration spaces
                forwardDiffeo(nextPtX, _thisDiffeo);

                //the transformation from the scaled demonstration space to the regular demonstration space
                _thisDiffeoDetails.undoTransform(nextPtX);
                //Invert the modifier
                if (_thisModifier){
                    _thisModifier->undoTransform(nextPtX);
                }
                //Done
            }
            //////////////////////////////////////////////
            template<typename M1, typename V1, typename V2>
            void getTraj( M1 & nextPtX, M1 & nextPtXd, const V1 & ptX, const V2 & tSteps ){
                //Take the given point and calculate the forward trajectory
                //Attention tSteps has to be in "relative time" -> the current position corresponds to t=0
                /*assert (nextPtX.rows() == _dim && "Wrong dimension");
                assert (nextPtX.cols() == tSteps.size() && "points and timesteps incompatible");
                assert (nextPtXd.cols() == tSteps.size() && "velocities and timesteps incompatible");
                */
                assert (ptX.size() == _dim && "Wrong dimension");

                //Apply target modifier if existent
                VectorXd ptPrime = ptX;
                if(_thisModifier){
                    _thisModifier->doTransform(ptPrime);
                }

                //Transform it into the scaled demonstration space
                _thisDiffeoDetails.doTransform(ptPrime);

                //Tranform the point into the control space
                reverseDiffeo(ptPrime, _thisDiffeo);

                //Check if finished
                _isFinished = (ptPrime.squaredNorm() <= _finishedNormSquare);

                //Integrate (explicit fixed step forward euler) in the transformed space
                //TBD nahhh change this
                double tCurr = 0.;
                int indCurr = 0;

                nextPtX = MatrixXd::Zero(_dim, tSteps.size());
                nextPtXd = MatrixXd::Zero(_dim, tSteps.size());

                int thisZone, thisZone2;
                MatrixXd thisA;

                double thisdT;
                VectorXd ptPrimed(_dim);
                //Initialise
                //cout << ptPrime << endl;
                thisZone = _fZone(ptPrime);
                //cout << thisZone << endl;
                //cout << _Alist[thisZone] << endl;
                //cout << *_Alist[thisZone] << endl;
                thisA = *_Alist[thisZone];
                while (tCurr < tSteps[tSteps.size()-1]){

                    if (tCurr+_dTint>tSteps[indCurr]){
                        //reaching a demanded timepoint
                        thisdT = tSteps[indCurr]-tCurr;
                        ptPrimed = thisA*ptPrime;
                        _scaleFunc(ptPrime, ptPrimed, _thisDiffeoDetails._controlSpaceVelocity, _breakTime, 1. );
                        //Integrate
                        ptPrime+=thisdT*ptPrimed;
                        //Save
                        //pos
                        nextPtX.col(indCurr) = ptPrime;
                        //vel
                        ptPrimed = thisA*ptPrime;
                        _scaleFunc(ptPrime, ptPrimed, _thisDiffeoDetails._controlSpaceVelocity, _breakTime, 1. );
                        nextPtXd.col(indCurr) = ptPrimed;
                        tCurr = tSteps(indCurr);
                        indCurr += 1;
                    }else{
                        //integrating
                        ptPrimed = thisA*ptPrime;
                        _scaleFunc(ptPrime, ptPrimed, _thisDiffeoDetails._controlSpaceVelocity, _breakTime, 1. );
                        //Integrate
                        ptPrime+=_dTint*ptPrimed;
                        tCurr+=_dTint;
                    }
                    thisZone2 = _fZone(ptPrime);
                    if(thisZone!=thisZone2){
                        thisZone = thisZone2;
                        thisA = *_Alist[thisZone];
                    }
                }

                //Transform the position and velocity from control to demonstration spaces
                forwardDiffeoVel(nextPtX, nextPtXd, _thisDiffeo);

                //the transformation from the scaled demonstration space to the regular demonstration space
                _thisDiffeoDetails.undoTransform(nextPtX, nextPtXd);
                //Invert the modifier
                if (_thisModifier){
                    _thisModifier->undoTransform(nextPtX, nextPtXd);
                }
                //Done
            }
            //////////////////////////////////////////////
            template<typename M1, typename V1, typename V2>
            void getTraj( M1 & nextPtX, M1 & nextPtXd, M1 & nextPtXdd, const V1 & ptX, const V2 & tSteps ){
                //Take the given point and calculate the forward trajectory
                //Attention tSteps has to be in "relative time" -> the current position corresponds to t=0
                /*assert (nextPtX.rows() == _dim && "Wrong dimension");
                assert (nextPtX.cols() == tSteps.size() && "points and timesteps incompatible");
                assert (nextPtXd.cols() == tSteps.size() && "velocities and timesteps incompatible");
                assert (nextPtXdd.cols() == tSteps.size() && "acceleration and timesteps incompatible");
                */
                assert (ptX.size() == _dim && "Wrong dimension");

                VectorXd tSteps2(2*tSteps.size());
                MatrixXd nextPtX2(_dim, 2*tSteps.size());
                MatrixXd nextPtXd2(_dim, 2*tSteps.size());

                for (size_t i=0; i<(size_t) tSteps.size(); ++i){
                    tSteps2(2*i)=tSteps(i);
                    tSteps2(2*i+1)=tSteps(i)+_dTnumericDiff;
                }

                //Apply target modifier if existent
                VectorXd ptPrime = ptX;
                if(_thisModifier){
                    _thisModifier->doTransform(ptPrime);
                }

                //Transform it into the scaled demonstration space
                _thisDiffeoDetails.doTransform(ptPrime);

                //Tranform the point into the control space
                reverseDiffeo(ptPrime, _thisDiffeo);

                //Check if finished
                _isFinished = (ptPrime.squaredNorm() <= _finishedNormSquare);

                //Integrate (explicit fixed step forward euler) in the transformed space
                //TBD nahhh change this
                double tCurr = 0.;
                int indCurr = 0;

                nextPtX = MatrixXd::Zero(_dim, tSteps.size());
                nextPtXd = MatrixXd::Zero(_dim, tSteps.size());
                nextPtXdd = MatrixXd::Zero(_dim, tSteps.size());

                int thisZone, thisZone2;
                MatrixXd thisA(_dim, _dim);

                double thisdT;
                VectorXd ptPrimed(_dim);
                //Initialise
                thisZone = _fZone(ptPrime);
                //cout << thisZone << endl;
                //cout << _Alist[thisZone] << endl;
                //cout << *_Alist[thisZone] << endl;
                thisA = *_Alist[thisZone];
                while (tCurr < tSteps2[tSteps2.size()-1]){

                    if (tCurr+_dTint>tSteps2[indCurr]){
                        //reaching a demanded timepoint
                        thisdT = tSteps2[indCurr]-tCurr;
                        ptPrimed = thisA*ptPrime;
                        _scaleFunc(ptPrime, ptPrimed, _thisDiffeoDetails._controlSpaceVelocity, _breakTime, 1. );
                        //Integrate
                        ptPrime+=thisdT*ptPrimed;
                        //Save
                        //pos
                        nextPtX2.col(indCurr) = ptPrime;
                        //vel
                        ptPrimed = thisA*ptPrime;
                        _scaleFunc(ptPrime, ptPrimed, _thisDiffeoDetails._controlSpaceVelocity, _breakTime, 1. );
                        nextPtXd2.col(indCurr) = ptPrimed;
                        tCurr = tSteps2(indCurr);
                        indCurr += 1;
                    }else{
                        //integrating
                        ptPrimed = thisA*ptPrime;
                        _scaleFunc(ptPrime, ptPrimed, _thisDiffeoDetails._controlSpaceVelocity, _breakTime, 1. );
                        //Integrate
                        ptPrime+=_dTint*ptPrimed;
                        tCurr+=_dTint;
                    }
                    thisZone2 = _fZone(ptPrime);
                    if(thisZone!=thisZone2){
                        thisZone = thisZone2;
                        thisA = *_Alist[thisZone];
                    }
                }

                //Transform the position and velocity from control to demonstration spaces
                forwardDiffeoVel(nextPtX2, nextPtXd2, _thisDiffeo);

                //Extract the demanded points and accelerations
                for (size_t i=0; i<(size_t) tSteps.size(); ++i){
                    nextPtX.col(i)=nextPtX2.col(2*i);
                    nextPtXd.col(i)=nextPtXd2.col(2*i);
                    nextPtXdd.col(i)=(nextPtXd2.col(2*i+1)-nextPtXd2.col(2*i))/_dTnumericDiff;
                }

                //the transformation from the scaled demonstration space to the regular demonstration space
                _thisDiffeoDetails.undoTransform(nextPtX, nextPtXd, nextPtXdd);
                //Invert the modifier
                if (_thisModifier){
                    _thisModifier->undoTransform(nextPtX, nextPtXd, nextPtXdd);
                }
                //Done
            }
            //////////////////////////////////////////////
    };

    //Simple state machine
    class simpleStateMachine{
        private:
            vector<bool *> _stateFinishedVec;
            vector<size_t> _stateNr;
            size_t _lastIdx;
            size_t _size;
        public:
            simpleStateMachine(){
                _lastIdx = 0;
            }
            ///////////
            void setNewState( bool * aBool, const size_t aNr){
                _stateFinishedVec.push_back(aBool);
                _stateNr.push_back(aNr);
                _size++;
            }
            ///////////
            const size_t operator()()const {
                return _stateNr[_lastIdx];
            }
            //////////
            void check(){
                if ( _stateFinishedVec[_lastIdx] && (_lastIdx < _size-1) ){
                    _lastIdx++;
                }
            }
            //////////
            void reset(){
                _lastIdx = 0;
            }
    };

    ///A class uniting multiple diffeomovements to replay movements not representable by a single diffeo
    class DiffeoMovement{
        private:
            vector<DiffeoMoveObj *> _allMoves;
            function<size_t()> _getStateFun;
            function<void()> _checkFun;

        public:
            ///////////////
            DiffeoMovement(){
                _allMoves.clear();
            }
            ///////////////
            void setMoveI(const size_t i, DiffeoMoveObj * moveIaddr){
                while(_allMoves.size()<=i){
                    _allMoves.push_back(nullptr);
                }
                _allMoves[i] = moveIaddr;
            }
            ///////////////
            void setGetStateFun(const function<size_t()> & newFun){
                _getStateFun = newFun;
            }
            ///////////////
            void setCheckFun(const function<void()> & newFun){
                _checkFun = newFun;
            }
            ///////////////
            template<typename M1, typename V1, typename V2>
            void getTraj( M1 & nextPtX, const V1 & ptX, const V2 & tSteps ){
                _allMoves[_getStateFun()]->getTraj( nextPtX, ptX, tSteps );
                _checkFun();
            }
            ///////////////
            template<typename M1, typename V1, typename V2>
            void getTraj( M1 & nextPtX, M1 & nextPtXd, const V1 & ptX, const V2 & tSteps ){
                _allMoves[_getStateFun()]->getTraj( nextPtX, nextPtXd, ptX, tSteps );
                _checkFun();
            }
            ///////////////
            template<typename M1, typename V1, typename V2>
            void getTraj( M1 & nextPtX, M1 & nextPtXd, M1 & nextPtXdd, const V1 & ptX, const V2 & tSteps ){
                _allMoves[_getStateFun()]->getTraj( nextPtX, nextPtXd, nextPtXdd, ptX, tSteps );
                _checkFun();
            }
    };


    bool searchDiffeoMovement(diffeoStruct& resultDiffeo, diffeoDetails& resultDetails, const MatrixXd & targetIn, const VectorXd & timeIn = VectorXd::Ones(1), const string & resultPath="", diffeoSearchOpts & opts= aDiffeoSearchOpt){
        //Search the geometric diffeo and set the additional informations concerning speed etc
        if (not( (targetIn.cols()==timeIn.size()) || (timeIn.size()==1) )){
            throw runtime_error("time.size is either 1 or target.cols");
        }

        //First make the demonstration end at 0 and scale it
        MatrixXd target = resultDetails.calcNtransform(targetIn, timeIn, opts);

        //Now get the source
        MatrixXd source = getSourceTraj(target, timeIn);

        //Now search for the diffeo
        if (not( iterativeSearch(source, target, resultDiffeo, opts) )){
            cout << "Search failed" << endl;
            return false;
        }

        //Save the results in file if demanded
        if (resultPath.compare("")!=0){
            resultDiffeo.toFolder(resultPath);
            resultDetails.toFolder(resultPath);
        }

        //Done
        return true;
    }
    ////////////////////////////////////////////
    bool searchDiffeoMovement(diffeoStruct& resultDiffeo, diffeoDetails& resultDetails, const string & inputPath, const string & resultPath="", const string & targetName = "thisTarget", const string & timeName="thisTime", diffeoSearchOpts & opts= aDiffeoSearchOpt){
        //Load data
        MatrixXd target = Leph::ReadMatrix(inputPath+targetName);
        VectorXd time;
        try{
            time = Leph::ReadVector(inputPath+timeName);
        }catch(...){
            cerr << "Could not read "+inputPath+timeName << endl;
            cerr << "Assuming no time given" << endl;
            time = VectorXd::Ones(1); time(0) = 1.;
        }
        //Call main function
        return searchDiffeoMovement(resultDiffeo, resultDetails, target, time, resultPath, aDiffeoSearchOpt);
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    //Modifiers
    //Adds a list of local modifiers
    class modifierDiffeo{
        public:
            DiffeoMoveObj * _aMovement;
            diffeoStruct * _aDiffeoStruct;
            int _numModifiers;
            int _initialNumTrans;
            int _dim;
            vector<VectorXd*> _modPoints;
            vector<VectorXd*> _modTransVects;
            vector<double> _modScaling;
            vector<double> _modCoeffs;

            modifierDiffeo( DiffeoMoveObj & origMove ){
                _aMovement = &origMove;
                _aDiffeoStruct = &origMove.getDiffeoStuct();
                _initialNumTrans = _aDiffeoStruct->numTrans;
                _dim = _aMovement->getDimension();
                _numModifiers = 0;
                _modPoints.clear();
                _modCoeffs.clear();
                _modScaling.clear();
                _modTransVects.clear();
            }
            //////
            void toInitial(){
                //Remove local modifiers that are appended at the end of the diffeoStruct
                _aDiffeoStruct->numTrans = _initialNumTrans;
                _aDiffeoStruct->centers.conservativeResize(_dim, _initialNumTrans);
                _aDiffeoStruct->targets.conservativeResize(_dim, _initialNumTrans);
                _aDiffeoStruct->coefs.conservativeResize(_initialNumTrans);
                _aDiffeoStruct->divisionCoefs.conservativeResize(_initialNumTrans);
            }
            //////
            void applyModification(){
                //Append the local modifiers
                _aDiffeoStruct->numTrans = _initialNumTrans + _numModifiers;
                _aDiffeoStruct->centers.conservativeResize(_dim, _initialNumTrans + _numModifiers);
                _aDiffeoStruct->targets.conservativeResize(_dim, _initialNumTrans + _numModifiers);
                _aDiffeoStruct->coefs.conservativeResize(_initialNumTrans + _numModifiers);
                _aDiffeoStruct->divisionCoefs.conservativeResize(_initialNumTrans + _numModifiers);

                //Compute the actual values
                for (unsigned i=0; i < _numModifiers; i++){
                    _aDiffeoStruct->centers.col(_initialNumTrans+i) = *(_modPoints[i]);
                    _aDiffeoStruct->targets.col(_initialNumTrans+i) = *(_modPoints[i])+_modScaling[i]*(*_modTransVects[i]);
                    _aDiffeoStruct->coefs(_initialNumTrans+i)=_modCoeffs[i];
                    _aDiffeoStruct->divisionCoefs(_initialNumTrans+i)=1.;
                    cout << i << endl;
                    cout << _aDiffeoStruct->centers.col(_initialNumTrans+i) << endl;
                    cout << _aDiffeoStruct->targets.col(_initialNumTrans+i) << endl;
                    //Check whether the added transformation was a diffeo
                    if (_modCoeffs[i] > 1./(exp(-1./2.)*sqrt(2.)*( _modScaling[i]*(*_modTransVects[i]).norm() ))){
                        cout << "The modifying transformation " << i << "is __ NOT __ diffeomorphic " << endl << _modCoeffs[i] << " to " << 1./(exp(-1./2.)*sqrt(2.)*( _modScaling[i]*(*_modTransVects[i]).norm() )) << endl;
                    }else{
                        cout << "The modifying transformation " << i << "is diffeomorphic " << endl << _modCoeffs[i] << " to " << 1./(exp(-1./2.)*sqrt(2.)*( _modScaling[i]*(*_modTransVects[i]).norm() )) << endl;
                    }
                }
            }
            //////
            void addCorrectionPair( const VectorXd & ptX, const VectorXd & corrDirec, double deltaT, double influenceOnPoint=0.35 ){
                //ptX: point in demonstration space forming the center point for the correction pair
                //corrDirec: displacement direction of the transformation
                //deltaT: Determines the influence zone
                //influenceOnPoint: the weight of the gaussian on the center point
                MatrixXd allPtX;
                MatrixXd allPtXd;
                MatrixXd allPtXdd;
                Vector2d allT;
                allT << 0.0, deltaT;
                _aMovement->getTraj(allPtX, allPtXd, allPtXdd
                , ptX, allT);
                VectorXd tangent = allPtXd.col(0);
                tangent.normalize();
                double dist = (allPtX.col(1)-allPtX.col(0)).norm();
                //Compute the centers of the transformations
                VectorXd * pointBefore = new VectorXd;
                *pointBefore = ptX-dist*tangent;
                cout << "Before P" << endl << *pointBefore << endl;
                VectorXd * pointAfter = new VectorXd;
                *pointAfter = ptX+dist*tangent;
                cout << "After P" << endl << *pointAfter << endl;
                //The drection
                VectorXd * dirBefore = new VectorXd;
                *dirBefore = -corrDirec;
                cout << "Before D" << endl << *dirBefore << endl;
                VectorXd * dirAfter = new VectorXd;
                cout << "After D" << endl << *dirAfter << endl;
                *dirAfter = corrDirec;
                //Attention the diffeo is applied onto !scaled! values, so
                VectorXd ptXprime=ptX;
                _aMovement->getDiffeoDetailsStruct().doTransform(ptXprime);
                _aMovement->getDiffeoDetailsStruct().doTransform(*pointBefore, *dirBefore);
                _aMovement->getDiffeoDetailsStruct().doTransform(*pointAfter, *dirAfter);
                cout << "Before P" << endl << *pointBefore << endl;
                cout << "After P" << endl << *pointAfter << endl;
                cout << "Before D" << endl << *dirBefore << endl;
                cout << "After D" << endl << *dirAfter << endl;
                dist = (ptXprime-(*pointAfter)).norm();
                //The coef
                double coef = -log(influenceOnPoint)/(dist*dist);

                //Store the shit
                _numModifiers += 2;
                _modPoints.push_back( pointBefore );
                _modPoints.push_back( pointAfter );
                _modTransVects.push_back( dirBefore );
                _modTransVects.push_back( dirAfter );
                _modCoeffs.push_back(coef);
                _modCoeffs.push_back(coef);
                _modScaling.push_back(0.);
                _modScaling.push_back(0.);
            }
    };
	////////////////////////////////////////////////////////////////////////////////////////
    DiffeoMoveObj searchDiffeoMovement(const MatrixXd & targetIn, const VectorXd & timeIn = VectorXd::Ones(1), const string & resultPath="", diffeoSearchOpts & opts= aDiffeoSearchOpt){
        //Create the output
        DiffeoMoveObj resultDiffeoObj = DiffeoMoveObj();
        diffeoStruct resultDiffeo;
        diffeoDetails resultDetails;
        //Call main function
        if (not searchDiffeoMovement(resultDiffeo, resultDetails, targetIn, timeIn, resultPath, opts)){
            throw runtime_error("Matching failed");
        };
        resultDiffeoObj.setDiffeoStruct(resultDiffeo);
        resultDiffeoObj.setDiffeoDetailsStruct(resultDetails);
        resultDiffeoObj.doInit();
        return resultDiffeoObj;
    }
	////////////////////////////////////////////////////////////////////////////////////////
    DiffeoMoveObj searchDiffeoMovement(const string & inputPath, const string & resultPath="", const string & targetName = "thisTarget", const string & timeName="thisTime", diffeoSearchOpts & opts= aDiffeoSearchOpt){
        //Load data
        MatrixXd target = Leph::ReadMatrix(inputPath+targetName);
        VectorXd time;
        try{
            time = Leph::ReadVector(inputPath+timeName);
        }catch(...){
            cerr << "Could not read "+inputPath+timeName << endl;
            cerr << "Assuming no time given" << endl;
            time = VectorXd::Ones(1); time(0) = 1.;
        }
        return searchDiffeoMovement(target, time, resultPath, opts);
    }
}

#endif // DIFFEO_MOVEMENTS_HPP
