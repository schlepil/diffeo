#ifndef DIFFEO_HPP
#define DIFFEO_HPP

#include <math.h>

#include <iostream>

#include <Eigen/Dense>

#include <string>

#include <functional>

#include <Eigen/StdVector>

//#include <Utils/FileVector.h>
//#include "/home/elfuius/ownCloud/thesis/Bordeaux/myStuff/newStuff/testDiffeo/fileMatrixVector.h"
#include "FileVector.h"

#ifndef PIExists
    #define PI 3.141592653589793
    #define Ang2Rad 3.141592653589793/180.
    #define Rad2Ang 180./3.141592653589793
    #define PIExists
#endif

using namespace std;
using namespace Eigen;

///////////////////////////////////////////////////////////////////////////////////

namespace DiffeoMove{
    ///////////////////////////////////////////////////////////////////////////////////

    /*
    *Structure keeping all information defining a diffeomorphism
    */
    struct diffeoStruct {
        MatrixXd centers; // centers and targets are column vectors stacked horizontally
        MatrixXd targets;
        VectorXd coefs; //coefs and division_coefs are column vectors
        VectorXd divisionCoefs;
        int numTrans;
    };
    ///////////////////////////////////////////////////////////////////////////////////
    /*
    *Function applying a diffeomorphic "forward" transformation
    *Denoting pt' the point after deformation we get
    *pt' = tau(pt) = pt + V.exp(-coef^2.||pt-center||_2^2)
    * and V = (target-center)/divisionCoef
    */
    template<typename V1, typename M1>
    inline void iterativeFunction(const V1 & center, const V1 & target, M1 & pt, const double divisionCoef, const double coef){

        assert(center.rows() == pt.rows() && "Points to be transformed do not have the same dimension as center and target");

        int M = pt.rows();
        int N = pt.cols();

        M1 result = coef*(pt.colwise() - center);
        result = -result.cwiseProduct(result);

        //I hope this can be done more efficiently
        M1 helper1 = ((target-center)/divisionCoef).replicate(1,N) ;
        M1 helper2 = result.colwise().sum().array().exp().replicate(M,1) ;

        pt += helper1.cwiseProduct(helper2);
    }
    //////////////////////
    /*
    *Same as iterative function but also calculating the jacobian of each point
    *As above the transformation is given as
    *pt' = tau(pt) = pt + V*exp(-coef^2.||pt-center||_2^2)
    *so
    *d/(d pt) tau(pt) = J_pt(tau) = Identity - 2*coef^2*exp(-coef^2.||pt-center||_2^2)*V.(pt-center)
    */
    template<typename vectorType, typename V1, typename M1>
    inline void iterativeFunctionJac(const V1 & center, const V1 & target, M1 & pt, const double divisionCoef, const double coef, vectorType & allJacs){

        assert(center.rows() == pt.rows() && "Points to be transformed do not have the same dimension as center and target");

        int M = pt.rows();
        int N = pt.cols();

        M1 result = coef*(pt.colwise() - center);
        result = -result.cwiseProduct(result);

        //I hope this can be done more efficiently
        M1 helper1 = ((target-center)/divisionCoef).replicate(1,N) ;
        M1 helper2 = result.colwise().sum().array().exp().replicate(M,1) ;

        //Calculate jacobians if necessary
        MatrixXd thisId = MatrixXd::Identity(M,M);
        Matrix<double,-1,1> V = helper1.col(0);
        M1 deltaPos = pt.colwise()-center;
        for (size_t i=0; i<N; i++){
            allJacs[i] = (thisId - helper2(0,i)*2.*coef*coef*V*( deltaPos.col(i).transpose() ))*(allJacs[i]);
        }

        pt += helper1.cwiseProduct(helper2);
    }
    //////////////////////
    /*
    *Same as iterative function but also transforming the velocity associated to each point
    *As above the transformation is given as
    *pt' = tau(pt) = pt + V*exp(-coef^2.||pt-center||_2^2)
    *so
    *d/(d pt) tau(pt) = J_pt(tau) = Identity - 2*coef^2*exp(-coef^2.||pt-center||_2^2)*V.(pt-center)
    */
    template<typename V1, typename M1>
    inline void iterativeFunctionVel(const V1 & center, const V1 & target, M1 & pt, M1 & vel, const double divisionCoef, const double coef){

        assert(center.rows() == pt.rows() && "Points to be transformed do not have the same dimension as center and target");
        assert(coef>0. && "coef needs to be positive");

        int M = pt.rows();
        int N = pt.cols();

        M1 result = coef*(pt.colwise() - center);
        result = -result.cwiseProduct(result);

        //I hope this can be done more efficiently
        M1 helper1 = ((target-center)/divisionCoef).replicate(1,N) ;
        M1 helper2 = result.colwise().sum().array().exp().replicate(M,1) ;

        //Calculate jacobians if necessary
        MatrixXd thisId = MatrixXd::Identity(M,M);
        Matrix<double,-1,1> V = helper1.col(0);
        M1 deltaPos = pt.colwise()-center;
        for (size_t i=0; i<N; i++){
            vel.col(i) = (thisId - helper2(0,i)*2.*coef*coef*V*( deltaPos.col(i).transpose() ))*vel.col(i);
        }

        pt += helper1.cwiseProduct(helper2);
    }


    ////////////////////////////////////////////////////////////////////////////////////
    /*A entire diffeo for multiple points
    The diffeomorphism tau is the composition of some diffeomorphic transformations of the form
    *pt' = tau_i(pt) = pt + V*exp(-coef^2.||pt-center||_2^2)
    *So
    *tau(pt) = tau_N(tau_(N-1)(...(tau_1(tau_0(pt)))))
    */
    template<typename M1>
    void forwardDiffeo(M1 & pt, const diffeoStruct & aDiffeo, int nStart=-1, int nStop=-1){

        int dim = pt.rows();
        int l = pt.cols();

        nStart = nStart<0 ? 0:nStart;
        nStop = nStop<0 ? aDiffeo.numTrans:nStop;
        //std::cout << "forward: " << nStart << "to: " << nStop << std::endl;
        //MatrixXd ptOut = pt;
        for (size_t i=nStart; i<nStop; i++){
            //ptOut = iterativeFunction(aDiffeo.centers.col(i), aDiffeo.targets.col(i), ptOut, aDiffeo.divisionCoefs(i), aDiffeo.coefs(i));
            iterativeFunction(aDiffeo.centers.col(i), aDiffeo.targets.col(i), pt, aDiffeo.divisionCoefs(i), aDiffeo.coefs(i)); //Inplace modification of pt and Jac
        }
        //return ptOut;
    }
    ////////////////////////////////////////////////////////////////////////////////////
    /*A entire diffeo for multiple points including jacobian calculation
    * tau^N = tau_N . tau_(N-1) . ... . tau_0 (where . denotes composition)
    *then
    *J_pt(tau) = J_(tau^(N-1)(pt))(tau_N)*...*J_(tau^1(pt))(tau_2)*J_(tau^0(pt))(tau_1)*J_pt(tau_0) where * denotes standard matrix product
    */
    template<typename vectorType, typename M1>
    void forwardDiffeoJac(M1 & pt, const diffeoStruct & aDiffeo, vectorType & allJacs, int nStart=-1, int nStop=-1){

        int dim = pt.rows();
        int l = pt.cols();

        nStart = nStart<0 ? 0:nStart;
        nStop = nStop<0 ? aDiffeo.numTrans:nStop;
        //std::cout << "forward: " << nStart << "to: " << nStop << std::endl;
        //MatrixXd ptOut = pt;

        //Initialize with identities
        allJacs.clear();
        for (size_t i=0; i<l; i++){
            allJacs.push_back(MatrixXd::Identity(dim,dim));
        }

        for (size_t i=nStart; i<nStop; i++){
            //ptOut = iterativeFunction(aDiffeo.centers.col(i), aDiffeo.targets.col(i), ptOut, aDiffeo.divisionCoefs(i), aDiffeo.coefs(i));
            iterativeFunctionJac(aDiffeo.centers.col(i), aDiffeo.targets.col(i), pt, aDiffeo.divisionCoefs(i), aDiffeo.coefs(i), allJacs); //Inplace modification of pt and Jac
        }
        //return ptOut;
    }
    ////////////////////////////////////////////////////////////////////////////////////
    /*An entire diffeo for multiple points transforming associated velocities
    * tau^N = tau_N . tau_(N-1) . ... . tau_0 (where . denotes composition)
    *then
    *J_pt(tau) = J_(tau^(N-1)(pt))(tau_N)*...*J_(tau^1(pt))(tau_2)*J_(tau^0(pt))(tau_1)*J_pt(tau_0) where * denotes standard matrix product
    */
    template<typename M1>
    void forwardDiffeoVel(M1 & pt, M1 & vel, const diffeoStruct & aDiffeo, int nStart=-1, int nStop=-1){

        int dim = pt.rows();
        int l = pt.cols();

        nStart = nStart<0 ? 0:nStart;
        nStop = nStop<0 ? aDiffeo.numTrans:nStop;
        //std::cout << "forward: " << nStart << "to: " << nStop << std::endl;
        //MatrixXd ptOut = pt;

        for (size_t i=nStart; i<nStop; i++){
            //ptOut = iterativeFunction(aDiffeo.centers.col(i), aDiffeo.targets.col(i), ptOut, aDiffeo.divisionCoefs(i), aDiffeo.coefs(i));
            iterativeFunctionVel(aDiffeo.centers.col(i), aDiffeo.targets.col(i), pt, vel, aDiffeo.divisionCoefs(i), aDiffeo.coefs(i)); //Inplace modification of pt and Jac
        }
        //return ptOut;
    }

    ////////////////////////////////////////////////////////////////////////////////////
    /*Helper
    *Inline function calculating the exponential factor of a transformation
    *V1 and V2 need to be different templates in order to allow the usage with
    */
    template<typename V1, typename V2>
    inline double radial(const V1 & center, const V2 & pt, const double coefSq){
        //V1 dX = pt-center;
        //return exp(-coefSq * ((double) dX.dot(dX)));
        return exp(-coefSq * ((double) (pt-center).squaredNorm()));
    }

    ////////////////////////////////////////////////////////////////////////////////////
    /*Computes the inverse transformation of tau
    *There exists no closed form solution for this problem, so we use a
    *bounded version of newton's method to find the
    *pt = inverse(tau)(pt')
    */
    template<typename MatorVec>
    void reverseDiffeo(MatorVec & pt, const diffeoStruct & aDiffeo, int nStart=-1, int nStop=-1, const double convCrit=1e-12){

        int dim = pt.rows();
        int nPoints = pt.cols();

        //assert((pt.cols()==1));

        nStart = nStart<0 ? aDiffeo.numTrans:nStart;
        nStop = nStop<0 ? 0:nStop;
        //std::cout << "reverse: " << nStart << "to: " << nStop << std::endl;

        VectorXd V(pt.size());//This translation
        VectorXd b(pt.size());
        //Matrix<double, -1,1> ptOut = pt;

        VectorXd thisCenter(pt.size());
        VectorXd thisTarget(pt.size());

        double lambda, g, deriv, errValue, r;
        double thisCoef, thisCoefSq;

        for (int indCol=0; indCol<nPoints; indCol++){
            for (int i=nStart-1; i>=nStop; i--){
                thisCenter = aDiffeo.centers.col(i);
                thisTarget = aDiffeo.targets.col(i);
                thisCoef = aDiffeo.coefs(i);
                thisCoefSq = thisCoef*thisCoef;

                lambda = 0.0;
                V = -(thisTarget-thisCenter)/aDiffeo.divisionCoefs(i);
                //b = ptOut-thisCenter;
                b = pt.col(indCol)-thisCenter;

                do{
                    r = radial(thisCenter, pt.col(indCol)+lambda*V, thisCoefSq);
                    g = 2.*V.dot(b+lambda*V);
                    deriv = -1-thisCoefSq*g*r;
                    errValue = r-lambda;
                    lambda = min(max(lambda-errValue/deriv, 0.), 1.);
                }while(abs(errValue)>convCrit);

                //ptOut = ptOut+lambda*V;
                pt.col(indCol) += lambda*V;
            }
        //return ptOut;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    /*Computes the inverse transformation of tau
    *There exists no closed form solution for this problem, so we use a
    *bounded version of newton's method to find the
    *pt = inverse(tau)(pt')
    *This method also provides jacobians associated to each point. By default the "forward" jacobians are returned, since there is no analytical form to directly obtain
    *the jacobians of the reverse transformation
    */
    template<typename MatorVec, typename vectorType>
    void reverseDiffeoJac(MatorVec & pt, vectorType & allJacs, const diffeoStruct & aDiffeo, int nStart=-1, int nStop=-1, const double convCrit=1e-12, const bool returnForwardJacs=true){

        int dim = pt.rows();
        int nPoints = pt.cols();

        //assert((pt.cols()==1));

        nStart = nStart<0 ? aDiffeo.numTrans:nStart;
        nStop = nStop<0 ? 0:nStop;
        //std::cout << "reverse: " << nStart << "to: " << nStop << std::endl;

        VectorXd V(pt.size());//This translation
        VectorXd b(pt.size());
        //Matrix<double, -1,1> ptOut = pt;

        VectorXd thisCenter(pt.size());
        VectorXd thisTarget(pt.size());

        double lambda, g, deriv, errValue, r;
        double thisCoef, thisCoefSq;


        MatrixXd thisIdentity = MatrixXd::Identity(dim,dim);
        allJacs.clear();
        for (size_t i=0; i<nPoints; i++){
            allJacs.push_back(thisIdentity);
        }

        for (int indCol=0; indCol<nPoints; indCol++){
            for (int i=nStart-1; i>=nStop; i--){
                thisCenter = aDiffeo.centers.col(i);
                thisTarget = aDiffeo.targets.col(i);
                thisCoef = aDiffeo.coefs(i);
                thisCoefSq = thisCoef*thisCoef;

                lambda = 0.0;
                V = -(thisTarget-thisCenter)/aDiffeo.divisionCoefs(i);
                //b = ptOut-thisCenter;
                b = pt.col(indCol)-thisCenter;

                do{
                    r = radial(thisCenter, pt.col(indCol)+lambda*V, thisCoefSq);
                    g = 2.*V.dot(b+lambda*V);
                    deriv = -1-thisCoefSq*g*r;
                    errValue = r-lambda;
                    lambda = min(max(lambda-errValue/deriv, 0.), 1.);
                }while(abs(errValue)>convCrit);

                //ptOut = ptOut+lambda*V;
                //Update the jacobian
                pt.col(indCol) += lambda*V;

                allJacs[indCol] = allJacs[indCol]*(thisIdentity + V.dot((pt.col(indCol)-thisCenter).transpose())*lambda*2.*thisCoefSq);
            }
            if (not returnForwardJacs){
                allJacs[indCol] = allJacs[indCol].inverse();
            }
        }
    }
//////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////
    /*Computes the inverse transformation of tau
    *There exists no closed form solution for this problem, so we use a
    *bounded version of newton's method to find the
    *pt = inverse(tau)(pt')
    *This method also transforms the velocities associated to each point from the demonstration to the control space
    */
    template<typename MatorVec>
    void reverseDiffeoVel(MatorVec & pt, MatorVec & vel, const diffeoStruct & aDiffeo, int nStart=-1, int nStop=-1, const double convCrit=1e-12){

        int dim = pt.rows();
        int nPoints = pt.cols();

        //assert((pt.cols()==1));

        nStart = nStart<0 ? aDiffeo.numTrans:nStart;
        nStop = nStop<0 ? 0:nStop;
        //std::cout << "reverse: " << nStart << "to: " << nStop << std::endl;

        VectorXd V(pt.size());//This translation
        VectorXd b(pt.size());
        //Matrix<double, -1,1> ptOut = pt;

        VectorXd thisCenter(pt.size());
        VectorXd thisTarget(pt.size());

        double lambda, g, deriv, errValue, r;
        double thisCoef, thisCoefSq;

        MatrixXd thisIdentity = MatrixXd::Identity(dim,dim);
        MatrixXd thisJacobian = thisIdentity;

        for (int indCol=0; indCol<nPoints; indCol++){
            thisJacobian = thisIdentity;
            for (int i=nStart-1; i>=nStop; i--){
                thisCenter = aDiffeo.centers.col(i);
                thisTarget = aDiffeo.targets.col(i);
                thisCoef = aDiffeo.coefs(i);
                thisCoefSq = thisCoef*thisCoef;

                lambda = 0.0;
                V = -(thisTarget-thisCenter)/aDiffeo.divisionCoefs(i);
                //b = ptOut-thisCenter;
                b = pt.col(indCol)-thisCenter;

                do{
                    r = radial(thisCenter, pt.col(indCol)+lambda*V, thisCoefSq);
                    g = 2.*V.dot(b+lambda*V);
                    deriv = -1-thisCoefSq*g*r;
                    errValue = r-lambda;
                    lambda = min(max(lambda-errValue/deriv, 0.), 1.);
                }while(abs(errValue)>convCrit);

                //ptOut = ptOut+lambda*V;
                //Update the jacobian
                pt.col(indCol) += lambda*V;

                thisJacobian = thisJacobian*(thisIdentity + V.dot((pt.col(indCol)-thisCenter).transpose())*lambda*2.*thisCoefSq);
            }
            vel.col(indCol) = thisJacobian.inverse()*vel.col(indCol);
        }
    }
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
//The main class and its helpers
//A funtor to determine the zone
//Return 0
struct ret0{
    int operator() (const VectorXd & ptX) const { return 0; };
};

struct myZoneFunc{
    int operator() (const VectorXd & ptX) const {
        double thisNorm = ptX.norm();
        if(thisNorm < 1.){
            return 0;
        } else{
            return 1;
        }
    };
};

//A functor to scale the velocity
/*
stepSize is the the norm of the of the vector (start point - final point) divided by the number of points in the demonstration.
If trying to obtain "exactly" the demonstration points set stepTime to (total demonstration time)/(number of demonstration points)
If trying to integrate the continuous time equation d/dt x = f(x) (so you actually want to obtain the velocity) set stepTime to 1.
If you want to get x' = x_0 + int_0^T (f(x)) then set stepTime to T
#TBD: Try to template the size to enhance perfs
*/
template<typename VM1>
struct standardScale{
    void operator () (const VM1 & ptY, VM1 & ptYd, const double & controlSpaceVelocity, const double & breakTime = .15, const double & stepTime = 1.) {

        VectorXd ptYnorm = ptY.colwise().norm();
        VectorXd thisVelocity(ptY.cols()) = VectorXd::Constant(ptY.cols(), controlSpaceVelocity);

        for (size_t i=0; i<(size_t) ptY.cols();++i){
            if( ptYNorm(i) < breakTime*controlSpaceVelocity ){
                thisVelocity(i) = thisVelocity(i)*ptYNorm(i)/(breakTime*controlSpaceVelocity);
                //std::cout << "stepSize: " << stepSize << " ; " << thisStepSize << std::endl;
            }
        }
        //Normalize the velocity with the desired velocity factor
        ptYd = stepTime * ( (ptYd.cwiseProduct( thisVelocity.replicate( ptY.rows(), 1 ) )).cwiseQuotient( ptYnorm.replicate(ptY.rows(), 1) ) );
    }
};

class DiffeoMoveObj{
    //friend KickMove;
    private:
        std::vector< MatrixXd* > _Alist;
        std::function<int (const VectorXd & ptX)> _fZone;
        std::function<void (const VectorXd & ptY, VectorXd & ptYd, const double & controlSpaceVelocity, const double & breakTime, const double & stepTime)> _scaleFunc;
        VectorXd _scaling;
        VectorXd _htOffset;
        MatrixXd _htRot;
        MatrixXd _htRotInv;
        MatrixXd _linGainRot;
        diffeoStruct _thisDiffeo;
        double _dTint;//Fixed explicit euler forward integration time-step
        double _controlSpaceVelocity;
        double _breakTime;

        bool _isInit;

    public:
        ///////////////////////////////////////////////
        DiffeoMoveObj(int dim = -1){

            _Alist.clear();
            _fZone = ret0();
            _isInit = false;
            _dim = dim;
        }
        //////////////////////////////////////////////
        void loadFolder(const std::string & aFolder)
        {
            //Replace standard values with values from files in folder if existing
            _thisDiffeo.centers = Leph::ReadMatrix( aFolder+"/centers.txt" );
            _thisDiffeo.targets = Leph::ReadMatrix( aFolder+"/targets.txt" );
            _thisDiffeo.coefs = Leph::ReadVector( aFolder+"/coefs.txt" );
            _thisDiffeo.divisionCoefs = Leph::ReadVector( aFolder+"/division_coefs.txt" );
            _thisDiffeo.numTrans = (int) Leph::ReadVector( aFolder+"/numTrans.txt")(1);

            //
            _scaling = Leph::ReadVector( aFolder+"/scaling.txt");
            _htOffset = Leph::ReadVector( aFolder+"/translation.txt" );
            _htRot = Leph::ReadMatrix( aFolder+"/rotationMatrix.txt" );
            _htRotInv = _htRot.inverse();
            //
            //Vector2d tmpVec = Leph::ReadVector( aFolder+"/stepSize.txt" );
            _controlSpaceVelocity = (Leph::ReadVector( aFolder+"/controlSpaceVelocity.txt" ))(1);

            //Get the rotation
            this->setLinGainRot(Leph::ReadMatrix( aFolder+"/rotationMatrixLinearGain.txt" ));

            assert( _thisDiffeo.centers.cols() == _thisDiffeo.targets.cols() && _thisDiffeo.centers.rows() == _thisDiffeo.targets.rows() && _thisDiffeo.numTrans == _thisDiffeo.centers.cols() && _thisDiffeo.numTrans == _thisDiffeo.targets.cols() );
            if ( not _isInit ){
                _dim = _thisDiffeo.centers.rows();
                doInit();
            }
            assert(_dim == _thisDiffeo.centers.rows() );
        }

        //////////////////////////////////////////////
        void doInit(){

            _scaleFunc = standardScale< Matrix<double, _dim, 1> >();
            _scaling = Matrix<double,_dim,1>::Constant(1.);
            _htOffset = Matrix<double,_dim,1>::Zero();
            _htRot = Matrix<double,_dim,_dim>::Identity();
            _htRotInv = Matrix<double,_dim,_dim>::Identity();
            _linGainRot = Matrix<double, _dim, _dim>::Identity();
            _dTint = 1e-3;
            _controlSpaceVelocity = -10000;

            _breakTime = 0.15;

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
        void setLinGainRot(MatrixXd newRot){

            MatrixXd linGainRotInv = _linGainRot.inverse();
            _linGainRot = newRot;
            for ( int i=0; i<_Alist.size(); i++){
                //Undo old rotation
                _Alist[i] = linGainRotInv.transpose()*(*_Alist[i])*linGainRotInv;
                //Apply new
                _Alist[i] = _linGainRot.transpose()*(*_Alist[i])*_linGainRot;
            }
        }
        //////////////////////////////////////////////
        void setControlSpaceVelocity(const double & newConstrolSpaceVelocity){
            assert(newConstrolSpaceVelocity>0. && "negative stepsize!");
            _constrolSpaceVelocity = newConstrolSpaceVelocity;
        }
        //////////////////////////////////////////////
        const double & getConstrolSpaceVelocity(){
            return _constrolSpaceVelocity;
        }
        //////////////////////////////////////////////
        void setBreakTime( const double & newBreakTime ){
            assert(newBreakTime>=0. & &"break time needs to be non-negative");
            _breakTime = newBreakTime;
        }
        //////////////////////////////////////////////
        const double & getBreakTime(){
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
        //////////////////////////////////////////////////////////////////
        template<typename Mat>
        void setZoneAi( const int i, const Mat & Ai ){
            //assert(i<nZones && "out of range");
            //cout << _linGainRot << endl;
            while (_Alist.size()<=i){
                Mat* thisMat = new Mat;
                *thisMat = Mat::Identity();
                _Alist.push_back( thisMat );
            }

            Mat* thisMat = new Mat;
            *thisMat = _linGainRot.transpose()*Ai*_linGainRot;
            _Alist[i] = thisMat;
        }
        //////////////////////////////////////////////
        void setZoneFunction( std::function<int (VectorXd pt)> fZone ){
            _fZone = fZone;
        }
        //////////////////////////////////////////////
        void setZoneScale( std::function<void ( const VectorXd & ptY, VectorXd & ptYd, const double & controlSpaceVelocity, const double & breakTime, const double & stepTime )> newScaleFunc ){
            _scaleFunc = newScaleFunc;
        }
        //////////////////////////////////////////////
        void setScaling(const VectorXd & newScaling){
            _scaling = newScaling;
        }
        //////////////////////////////////////////////
        void sethtOffset(const VectorXd & newOffset){
            _htOffset = newOffset;
        }
        //////////////////////////////////////////////
        void sethtRot( const MatrixXd & newRot){
            //Check if rotqtion matrix?
            _htRot = newRot;
            _htRotInv = newRot.inverse();
        }
        //////////////////////////////////////////////
        void sethtRotInv( const MatrixXd & newRotInv){
            //Check if rotqtion matrix?
            _htRotInv = newRotInv;
            _htRot = newRotInv.inverse();
        }
        //////////////////////////////////////////////
        template<typename V1>
        V1 applyForward( const V1 & ptY ){
            V1 ptX = ptY;

            forwardDiffeo(ptX, _thisDiffeo);
            //Invert the preliminary transformation
            //Invert scaling
            ptX = ptX.cwiseProduct(_scaling);

            ptX = _htRotInv*ptX-_htOffset;

            return ptX;
        }
        //////////////////////////////////////////////
        template<typename V1>
        V1 applyReverse( const V1 & ptX ){
            //Perform preliminary transformation
            V1 ptY = _htRot*(ptX+_htOffset);
            //Scale it
            ptY = ptY.cwiseQuotient(_scaling);
            reverseDiffeo(ptY, _thisDiffeo);
            return ptY;
        }
        //////////////////////////////////////////////
        template<typename M1, typename V1, typename V2>
        void getTraj( M1 & nextPtX, M1 & nextPtXd, const V1 & ptX, const V2 & tSteps ){
            //Take the given point and calculate the forward trajectory
            //Attention tSteps has to be in "relative time" -> the current position corresponds to t=0
            assert (nextPtX.rows() == _dim & &"Wrong dimension");
            assert (nextPtX.cols() == tSteps.size() & &"points and timesteps incompatible");
            assert (ptX.size() == _dim & &"Wrong dimension");


            int thisZone;
            MatrixXd thisA(_dim, _dim);

            //Perform preliminary transformation
            VectorXd ptPrime(_dim);
            ptPrime = _htRot*(ptX+_htOffset);
            //Scale it
            ptPrime = ptPrime.cwiseQuotient(_scaling);
            //Apply inverse transformation
            reverseDiffeo(ptPrime, _thisDiffeo);
            //cout << ptPrime.norm() << endl;
            //Integrate (explicit fixed step forward euler) in the transformed space
            double tCurr = 0.;
            int indCurr = 0;

            //cout << _linGainRot*ptPrime << endl;

            nextPtX = MatrixXd::Zero(_dim, tSteps.size());
            nextPtXd = MatrixXd::Zero(_dim, tSteps.size());

            VectorXd ptPrimed(_dim);
            double thisdT;


            while (tCurr < tSteps[tSteps.size()-1]){
                thisZone = _fZone(ptX);
                thisA = *_Alist[thisZone];

                if (tCurr+_dTint>tSteps[indCurr]){
                    //reaching a demanded timepoint
                    thisdT = tSteps[indCurr]-tCurr;
                    ptPrimed = thisA*ptPrime;
                    _scaleFunc(ptPrime, ptPrimed, _controlSpaceVelocity, _breakTime, 1. );
                    //Integrate
                    ptPrime+=thisdT*ptPrimed;
                    //Save
                    //pos
                    nextPtX.col(indCurr) = ptPrime;
                    //vel
                    ptPrimed = thisA*ptPrime;
                    _scaleFunc(ptPrime, ptPrimed, _stepSize);
                    nextPtXd.col(indCurr) = ptPrimed;
                    tCurr = tSteps(indCurr);
                    indCurr += 1;
                }else{
                    //integrating
                    ptPrimed = thisA*ptPrime;
                    _scaleFunc(ptPrime, ptPrimed, _controlSpaceVelocity, _breakTime, 1. );
                    //Integrate
                    ptPrime+=_dTint*ptPrimed;
                    tCurr+=_dTint;
                }
            }

            //Transform the position and velocity from control to demonstration spaces
            forwardDiffeoVel(nextPtX, nextPtXd, _thisDiffeo);

            //Invert the preliminary transformation
            //Invert scaling
            //#TBD is this the best way?
            for (int i=0; i<tSteps.size(); i++){
                nextPtX.col(i) = nextPtX.col(i).cwiseProduct(_scaling);
                nextPtXd.col(i) = nextPtXd.col(i).cwiseProduct(_scaling);
            }
            //cout << nextPtX << endl;
            nextPtX = (_htRotInv*nextPtX).colwise()-_htOffset;
            //Velocity and needs to be rotated
            nextPtXd = _htRotInv*nextPtXd;
        }
        //////////////////////////////////////////////
};

///////////////////////////////////////////////////////////////
//helpers

/*Matrix3d rotZ(double angInDeg){
    Matrix3d m;
    m = AngleAxisd(angInDeg*M_PI/180., Vector3d::UnitZ());
    return m;
}*/
///////////////////////////////////////////////////////////////
class movementModifier{
    private:

        int _dim;

        VectorXd _baseOffset;
        VectorXd _relatifOffset;
        VectorXd _offset;

        MatrixXd _baseRotation;
        MatrixXd _relatifRotation;
        MatrixXd _rotation;

        MatrixXd _baseRotationInv;
        MatrixXd _relatifRotationInv;
        MatrixXd _rotationInv;

    public:
        double alphaTranslation;
        double alphaRotation;

        movementModifier(const int dim){
            _dim = dim;

            _anchorPoint = VectorXd::Zero(dim);
            _offset = VectorXd::Zero(dim);

            _rotation = MatrixXd::Identity(dim);

            _rotationInv = MatrixXd::Identity(dim);
        }
        ///////////////////////////////////////////////////
        void setAnchorPoint(const VectorXd & newAnchorPoint){
            assert(_dim==newAnchorPoint.size() & &"Offset has wrong dimension");
            _anchorPoint = newAnchorPoint;
        }
        ///////////////////////////////////////////////////
        void setOffset(const VectorXd & newOffset){
            assert(_dim==newOffset.size() & &"Offset has wrong dimension");
            _offset = newOffset;
        }
        ///////////////////////////////////////////////////
        void setRotation(const MatrixXd & newRotation){
            assert(newRotation.cols() == newRotation.rows() & &"Rotation matrix is not square");
            assert(newRotation.cols() == _dim & &"Rotation matrix has wrong dimension");

            _rotation = newRotation;
            _rotationInv = _rotation.transpose();
        }
        ///////////////////////////////////////////////////
        inline void preTransFormPos( MatrixXd & ptX ){
            //Apply the preliminary transformation
            //The preliminary transformation corresponds to a rotation around the baseoffset plus a relatif offset
            ptX = (_rotation*(ptX.colwise()-_anchorPoint)).colwise()+_anchorPoint;
            ptX = ptX.colwise() + _offset;
        }
        ///////////////////////////////////////////////////
        inline void postTransFormPos(MatrixXd & ptX){
            ptX = ptX.colwise() - _offset;
            ptX = (_rotationInv*(ptX.colwise()-_anchorPoint)).colwise() + _anchorPoint;
        }
};
///////////////////////////////////////////////////////////////
class stateMachine{
    private:
        int _lastState;

    public:
        virtual void getNextState(const VectorXd & ptY);

        inline void setLastState(const int newLastState){
            assert (newLastState > 0 & &"No negative state ids allowed");
            _lastState = newLastState;
        }
};
///////////////////////////////////////////////////////////////
class succesifStates : public stateMachine{
    private:
        double _initialDist;
        double _transitionDist;
        double _currentRelPos;
        bool _new;
    public:

        const double & curRelPos; //Emulating read-only var
        const int & curState;

        succesifStates() : currentRelPos(_currentRelPos), curState(_lastState){
            _lastState = -1;
            _initialDist = -1.;
            _transitionDist = 0.1;
            _currentRelPos = 1.;
            _new = true;
        }
        ///////////////////////////////////////////////////////////////
        inline void setTransDist(const double & newTransDist){
            assert( newTransDist>0. & &"Transition distance must be strictly positive");
            _transitionDist = newTransDist;
        }
        ///////////////////////////////////////////////////////////////
        inline void setInitialDist(const double & newInitialDist){
            assert( newInitialDist>0. & &"Initial distance must be strictly positive");
            double oldRelDist = _initialDist*_currentRelPos;
            _initialDist = newInitialDist;
            _currentRelPos = oldRelDist/_initialDist;
        }
        ///////////////////////////////////////////////////////////////
        const int getNextState(const VectorXd & ptY){
            double thisNorm = ptY.norm();

            if (_new){
                _initialDist = thisNorm;
                _new = false;
            }

            _currentRelPos = thisNorm/_initialDist;

            if(thisNorm < _transitionDist){
                _lastState += 1;
                _currentRelPos = 1.;
                _new = true;
            }
            return _lastState;
        }
        ///////////////////////////////////////////////////////////////
};
///////////////////////////////////////////////////////////////
class diffeoMotion{
    private:

        int _dim;

        bool _isInit;

        std::vector<DiffeoMoveObj*> _moveList;

    public:

        stateMachine* thisStateMachine;
        movementModifier* thisModifier;

        KickMove(){



            _lastState = 0;
            _nextState = 0;
            _goalAngle = 0.0;
            _dim=-1;
            _ballPosOffset = Matrix<double, dim, 1>::Zero();
            _idealBallPos = Matrix<double, dim, 1>::Zero();
            _initialBackNorm = 0.;

            this->setIdealBallPos(0.03, -0.11);

            _moveList.clear();

            _isInit = false;
        }

        ///////////////////////////////////////////////////////////
        void doInit(){
            assert (backwardMoveObj.getDim() == forwardMove.getDim() & &"Backward and forwaard move do not have the same dimension");
            assert (backwardMoveObj.isInit() & forwardMoveObj.isInit() & &"Forward or backward move are not initialised");

            _dim = backwardMoveObj.getDim();

        }
        ///////////////////////////////////////////////////////////
        void setMovement( const int i, const DiffeoMoveObj & moveI ){

            while (_moveList.size()<=i){
                DiffeoMoveObj* thisMove = new DiffeoMoveObj;
                _moveList.push_back( thisMove );
            }
            _moveList[i] = &newMoveI;
        }
        ///////////////////////////////////////////////////////////
        void setStateFunc( std::function<int(const int lastState, const VectorXd & currentPos)> newStateFun ){
            _nextStateFun = newStateFun;
        }
        ///////////////////////////////////////////////////////////
        void setLastState( const int newOldState ){
            _lastState = newOldState;
        }
        ///////////////////////////////////////////////////////////
        const int getState(){
            return _lastState;
        }
        ///////////////////////////////////////////////////////////
        void setBallOffset(const double & newXoff, const double & newYoff){
            _ballPosOffset(0) = -newXoff; //The ball offset already has inversed sign
            _ballPosOffset(1) = -newYoff;
            return void();
        }
        ///////////////////////////////////////////////////////////
        void setBallPos(const double & newX, const double & newY){
            _ballPosOffset(0) = -(newX-_idealBallPos(0));
            _ballPosOffset(1) = -(newY-_idealBallPos(1));
            return void();
        }
        ///////////////////////////////////////////////////////////
        void setIdealBallPos(const double & newX, const double & newY){
            //Correct offset
            _ballPosOffset(0) -= (newX-_idealBallPos(0));
            _ballPosOffset(1) -= (newX-_idealBallPos(1));
            //Set new pos
            _idealBallPos(0) = newX;
            _idealBallPos(1) = newY;
            return void();
        }
        ///////////////////////////////////////////////////////////
        void setGoalAngle(const double & _newGoalAng){
            _goalAngle = _newGoalAng;
        }
        ///////////////////////////////////////////////////////////
        //Skip all the ball parameters now
        ///////////////////////////////////////////////////////////
        void getStep(Matrix<double,dim,1> & nextPt, Matrix<double,dim,1> & nextPtd, Matrix<double,dim,1> & nextPtdd, const double & nextT, const Matrix<double,dim,1> & ptX, const double & numDiffDt = 1e-4){
            //Get

            //Do some stuff on ball positioning

            //Get the next state
            Matrix<double,dim,3> nextPtAll = Matrix<double,dim,3>::Zero();
            Matrix<double,dim,3> nextPtdAll = Matrix<double,dim,3>::Zero();
            Matrix<double,3,1> allT = Matrix<double,3,1>::Zero();
            allT << nextT-numDiffDt, nextT, nextT+numDiffDt;

            //Apply precorrection due to balloffset / desired angle
            //cout << ptX << endl << endl;
            Matrix<double,dim,1> ptXCorr = ptX + _ballPosOffset;

            //Perform desired rotation AROUND the ball
            Matrix<double, dim, dim> thisRot = Matrix<double, dim, dim>::Identity();
            thisRot(0,0) = cos(-_goalAngle*Ang2Rad);
            thisRot(1,1) = thisRot(0,0);
            thisRot(0,1) = -sin(-_goalAngle*Ang2Rad);
            thisRot(1,0) = -thisRot(0,1);
            Matrix<double, dim, dim> thisRotInv = thisRot.transpose();//Inverse of a rotation matrix is its transpose

            //cout << thisRot << endl << endl;

            //Shift -> Rotate -> undo shift
            ptXCorr = thisRot*(ptXCorr-_idealBallPos);
            ptXCorr += _idealBallPos;
            //take care of foot orientation
            //footPos; trunkPos; footAxis; trunkAxis -> footZ <=> 8
            ptXCorr(8) -= _goalAngle*Ang2Rad;

            //cout << ptXCorr << endl << endl;

            //Simple strategy to determine the next state
            Matrix<double,dim,1> ptY;
            _nextState = _lastState;
            if (_lastState == -1){
                ptY = _backwardMove.applyReverse(ptXCorr);
                //When close -> Done
                double thisNorm = ptY.norm();
                if (thisNorm < 0.5*_initialBackNorm){
                    //Reduce the offset when reaching the end
                    _ballPosOffset *= thisNorm/(0.5*_initialBackNorm);
                }
                if (thisNorm <= 0.015*0.5*_backwardMove.getStepSize()){
                    _nextState = 0;
                }
            }else if( _lastState == 1 ){
                ptY = _forwardMove.applyReverse(ptXCorr);
                //cout << ptY.norm() << " ; " << 0.015*3.*_forwardMove.getStepSize() << endl;
                if (ptY.norm() <= 0.015*3.*_forwardMove.getStepSize()){
                    _nextState = -1;
                    //safe the norm
                    _initialBackNorm = _backwardMove.applyReverse(ptXCorr).norm();
                }
            }
            //cout << ptY << endl << endl;

            if (_nextState==0){
                //Default state for not moving
                /*
                if ((ptX-_goToPos).norm() < 0.01){
                    nextPt = _goToPos;
                    cout << "reached" << endl;
                }else{
                    nextPt = 0.98*ptX+0.02*_goToPos;
                    cout << "approaching" << endl;
                }

                nextPt = (0.99*ptX)+(0.01*_goToPos);
                //nextPt = _goToPos;//ptX;
                nextPtd = Matrix<double,dim,1>::Zero();
                nextPtdd = Matrix<double,dim,1>::Zero();
                return void();
                */
                //_backwardMove.getTraj(nextPtAll, nextPtdAll, ptXCorr, allT);//<Matrix<double,dim,3> >
                //nextPtdAll = Matrix<double,dim,3>::Zero();

                nextPt = ptX;
                nextPtd = Matrix<double,dim,1>::Zero();
                nextPtdd = Matrix<double,dim,1>::Zero();
                return void();

            }else if(_nextState==-1){
                //Backward
                _backwardMove.getTraj(nextPtAll, nextPtdAll, ptXCorr, allT);//<Matrix<double,dim,3> >
            }else if(_nextState==1){
                //Forward
                _forwardMove.getTraj(nextPtAll, nextPtdAll, ptXCorr, allT);//<Matrix<double,dim,3> >
            }
            //Calc
            //undo precorrection due to desired angle
            //foot angle
            //This is extremely annoying
            //cout << nextPtAll << endl << endl;
            for (size_t i=0; i<nextPtAll.cols(); i++){
                nextPtAll(8,i) += _goalAngle*Ang2Rad;
            }
            //nextPtAll.row(8) = nextPtAll.row(8) + _goalAngle*Ang2Rad;

            //Shift -> undoRot -> undoshift
            nextPtAll = thisRotInv*(nextPtAll.colwise()-_idealBallPos);
            nextPtAll = nextPtAll.colwise()+_idealBallPos;
            //cout << nextPtAll << endl << endl;
            //"Undo" the velocity rotation
            nextPtdAll = thisRotInv*nextPtdAll;

            //undo precorrection due to desired angle
            nextPtAll = nextPtAll.colwise() - _ballPosOffset;

            nextPt = nextPtAll.col(1);
            nextPtd = nextPtdAll.col(1);
            nextPtdd = ( nextPtdAll.col(2)-nextPtdAll.col(0) )/(2.*numDiffDt);

            _lastState = _nextState;
            return void();
        }

};

}
#endif
