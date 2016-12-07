#ifndef DIFFEO_METHODS_HPP
#define DIFFEO_METHODS_HPP

#include<iostream>
#include <math.h>

#include <Eigen/Core>

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

namespace DiffeoMethods{
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
        ///////////////////////////////////////////////////////////
        bool toFolder(const string & folder, const string writeMode = "cpp"){

            if (not ( writeMode.compare("cpp")==0 || writeMode.compare("python")==0)){
                std::cout << "writeMode is either cpp or python" << endl;
                return false;
            }

            VectorXd thisNumTrans(1);
            thisNumTrans(0) = numTrans;

            Leph::WriteMatrix(folder+"diffeo/centers.txt", centers, 32, writeMode);
            Leph::WriteMatrix(folder+"diffeo/targets.txt", targets, 32, writeMode);
            Leph::WriteVector(folder+"diffeo/coefs.txt", coefs, 32, writeMode);
            Leph::WriteVector(folder+"diffeo/divisionCoefs.txt", divisionCoefs, 32, writeMode);
            Leph::WriteVector(folder+"diffeo/numTrans.txt", thisNumTrans, 32, writeMode);

            return true;
        }
        ///////////////////////////////////////////////////////////
        bool fromFolder(const string & folder){
        try{
            centers = Leph::ReadMatrix(folder+"diffeo/centers.txt");
            targets = Leph::ReadMatrix(folder+"diffeo/targets.txt");
            coefs = Leph::ReadVector(folder+"diffeo/coefs.txt");
            divisionCoefs = Leph::ReadVector(folder+"diffeo/divisionCoefs.txt");
            numTrans = (int) (Leph::ReadVector(folder+"diffeo/numTrans.txt")) (0);

            if (not(centers.cols()==numTrans && targets.cols()==numTrans && coefs.size()==numTrans && divisionCoefs.size()==numTrans)){
                std::cerr << "Inconsistent diffeo in number of transitions" << endl;
                return false;
            }
            if (not( centers.rows()==targets.rows() )){
                std::cerr << "Inconsistent diffeo in center and target dimension" << endl;
            }

        }catch(const std::exception &excp){
            std::cerr << excp.what() << endl;
            return false;
        }catch(...){
            std::cerr << "Unknown exception" << endl;
            return false;
        }
        return true;
        }
    };

    ///////////////////////////////////////////////////////////////////////////////////
    /*
    *Function applying a diffeomorphic "forward" transformation
    *Denoting pt' the point after deformation we get
    *pt' = tau(pt) = pt + V.exp(-coef^2.||pt-center||_2^2)
    * and V = (target-center)/divisionCoef
    */
    //#TBD schlepil check if && aka rvalue reference behaves as expected
    template<typename V1, typename V2, typename M1>
    inline void iterativeFunction(const V1 & center, const V2 & target, M1 & pt, const double & divisionCoef, const double & coef){

        assert(center.rows() == pt.rows() && "Points to be transformed do not have the same dimension as center and target");
        assert(center.rows() == target.rows() && "Inconsistent");
        assert(center.cols() == 1 && target.cols()==1 && "Inconsistent");

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
        //Test
        //pt += (((((target-center)/divisionCoef).replicate(1,N))).cwise(((result.colwise().sum().array().exp().replicate(M,1)))))
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

        size_t M = pt.rows();
        size_t N = pt.cols();

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

        nStart = nStart<0 ? 0:nStart;
        nStop = nStop<0 ? aDiffeo.numTrans:nStop;
        //std::cout << "forward: " << nStart << "to: " << nStop << std::endl;
        //MatrixXd ptOut = pt;
        size_t i;
        for (i=(size_t) nStart; i<(size_t) nStop; ++i){
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
        nStart = nStart<0 ? 0:nStart;
        nStop = nStop<0 ? aDiffeo.numTrans:nStop;
        //std::cout << "forward: " << nStart << "to: " << nStop << std::endl;
        //MatrixXd ptOut = pt;

        for (size_t i=(size_t)nStart; i<(size_t)nStop; i++){
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

        size_t nPoints = pt.cols();

        //assert((pt.cols()==1));

        nStart = nStart<0 ? aDiffeo.numTrans:nStart;
        nStop = nStop<0 ? 0:nStop;
        //std::cout << "reverse: " << nStart << "to: " << nStop << std::endl;

        VectorXd V(pt.rows());//This translation
        VectorXd b(pt.rows());
        //Matrix<double, -1,1> ptOut = pt;

        VectorXd thisCenter(pt.rows());
        VectorXd thisTarget(pt.rows());

        double lambda, g, deriv, errValue, r;
        double thisCoef, thisCoefSq;

        //Parallelize for each point?
        for (size_t indCol=0; indCol<nPoints; indCol++){
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

                allJacs[indCol] = allJacs[indCol]*(thisIdentity + V*((pt.col(indCol)-thisCenter).transpose())*lambda*2.*thisCoefSq);
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

                thisJacobian = thisJacobian*(thisIdentity + V*(((pt.col(indCol)-thisCenter).transpose()))*lambda*2.*thisCoefSq);
            }
            vel.col(indCol) = thisJacobian.inverse()*vel.col(indCol);
        }
    }
}; // end namespace
#endif
