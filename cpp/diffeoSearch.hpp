#ifndef DIFFEO_SEARCH_HPP
#define DIFFEO_SEARCH_HPP

#include <math.h>

#include "diffeoMethods.hpp"
#include "thingsThatShouldExist.hpp"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;
using namespace thingsThatShouldExist;

namespace DiffeoMethods{

    /*
    each: each non-degenerate dimension has variance one after scaling
    minimal: The scaling factor is s.t. the dimension with the SMALLEST variance has variance one after scaling. This factor is applied to all dimensions
    maximal: The scaling factor is s.t. the dimension with the LARGEST variance has variance one after scaling. This factor is applied to all dimensions
    */
    enum scalingMethod { each, minimal, maximal };
    ///////////////////////////////////////////////
    class diffeoSearchOpts{

        private:
            bool _isUpToDate;
            int _numIterations;
            VectorXi _points;
            VectorXd _divisionCoefList, _safeCoefList;
            VectorXd _divisionCoefs, _safeCoefs;

        public:
            double maxCoef, convCrit, regularise;
            const int & numIterations;
            const bool & isUpToDate;
            scalingMethod thisScalingMethod;

            diffeoSearchOpts(const int numIterationsIn=150):numIterations(_numIterations), isUpToDate(_isUpToDate){
                if (numIterationsIn<=0){
                    throw;
                }
                _isUpToDate=false;
                _numIterations = numIterationsIn;

                //Default values
                maxCoef = 10000.;//5.;
                convCrit = 1e-6;
                regularise = .1e-3;


                _points = VectorXd::LinSpaced(7, 0, _numIterations-1).cast<int>();
                //This should not be necessary but somehow 149 (as double) is cast to 148 (as int)
                _points(0)=0;
                _points(_points.size()-1)=_numIterations-1;
                cout << _points << endl;
                _divisionCoefList = VectorXd::Zero(7);
                _divisionCoefList << 3,3,2.5,2.0,1.66,1.33,1.1;//3.,2.5,2,1.5,1.25,1.2;
                _safeCoefList = VectorXd::Zero(7);
                _safeCoefList << 0.9, 0.9, 0.9, 0.85, 0.8, 0.75, 0.75;//0.8, 0.8, 0.8, 0.8, 0.6, 0.6;

                thisScalingMethod = each;

                this->doInterpolate();
            }
            /////////////////////////////////////////////////
            void setAnchorPoints(const VectorXi & newPoints){
                _points.resize(newPoints.size());
                _points = newPoints;
                this->setNumTrans( (int)_points(_points.size()-1) );
            }
            /////////////////////////////////////////////////
            void setAnchorPoints(VectorXd & newPoints){
                newPoints *= ((double) _numIterations) / ((double) newPoints(newPoints.size()-1));
                _points.resize(newPoints.size());
                this->setNumTrans( (int)_points(_points.size()-1) );
            }
            /////////////////////////////////////////////////
            void setNumTrans(const int newNumTrans){
                if (newNumTrans<=0){
                    throw;
                }
                _numIterations = newNumTrans;
                int L = _points.size();
                _points = VectorXi::LinSpaced(L, 0, _numIterations-1);
                //This should not be necessary but somehow 149 (as double) is cast to 148 (as int)
                _points(0)=0;
                _points(_points.size()-1)=_numIterations-1;
                this->doInterpolate();
            }
            /////////////////////////////////////////////////
            void setDivisionCoefList(const VectorXd & newDivisionCoefList){
                _divisionCoefList.resize(newDivisionCoefList.size());
                _divisionCoefList = newDivisionCoefList;
                this->doInterpolate();
            }
            /////////////////////////////////////////////////
            void setSafeCoefList(const VectorXd & newSafeCoefList){
                _safeCoefs.resize(newSafeCoefList.size());
                _safeCoefList = newSafeCoefList;
                this->doInterpolate();
            }
            /////////////////////////////////////////////////
            void doInterpolate(){
                /*cout << _points << endl;
                cout << _points(_points.size()-1) << endl;
                cout << _numIterations << endl;*/

                if ( _points.size()== _divisionCoefList.size() && _points.size()== _safeCoefList.size() && _points(_points.size()-1)==_numIterations-1 ){
                    _divisionCoefs.resize(_numIterations);
                    _safeCoefs.resize(_numIterations);
                    _divisionCoefs = VectorXd::Zero(_numIterations);
                    _safeCoefs = VectorXd::Zero(_numIterations);

                    //cout << _divisionCoefList << endl;
                    //cout << _safeCoefList << endl;

                    int k = 0;
                    for (int i=0; i<_numIterations; ++i){
                        if ( i>_points(k+1) ){
                            k++;
                        }

                        /*cout << _points(k) << endl;
                        cout << _points(k+1) << endl;
                        cout << _divisionCoefs(i) << endl;*/
                        //cout << (i-_points(k))/((double)(_points(k+1)-_points(k))) <<endl;

                        _divisionCoefs(i) = _divisionCoefList(k) + (_divisionCoefList(k+1)-_divisionCoefList(k))*(i-_points(k))/((double)(_points(k+1)-_points(k)));
                        _safeCoefs(i) = _safeCoefList(k) + (_safeCoefList(k+1)-_safeCoefList(k))*(i-_points(k))/(_points(k+1)-_points(k));

                        //cout << _divisionCoefs(i) << " " << _safeCoefs(i) << endl;
                    }
                    _isUpToDate=true;
                }else{
                    _isUpToDate=false;
                    _divisionCoefs.resize(0);
                    _safeCoefs.resize(0);
                    _divisionCoefs = VectorXd::Zero(0);
                    _safeCoefs = VectorXd::Zero(0);
                }
            }
            /////////////////////////////////////////////////
            bool getDivisionNSafe(VectorXd & division, VectorXd & safe) const {
                division = _divisionCoefs;
                safe = _safeCoefs;
                return _isUpToDate;
            }
    };
    //Get a dummy
    diffeoSearchOpts aDiffeoSearchOpt;


    /////////////////////////////////////////////
    MatrixXd getSourceTraj(const MatrixXd & target, const VectorXd & timeIn = VectorXd::Ones(1)){

        int nPoints = target.cols();
        int dim = target.rows();

        MatrixXd time(1,nPoints);

        if(timeIn.size()==1){
            time.topRows<1>().setLinSpaced(nPoints, 1.,0);
        }else if (time.size() == nPoints){
            time.topRows<1>() = timeIn/timeIn(0);
        }else{
            throw runtime_error("time must either have one element or have the same size as target.cols");
        }

        MatrixXd source = (time.replicate(dim,1)).cwiseProduct( (target.leftCols<1>().replicate(1,nPoints)) );

        return source;
    }
    /////////////////////////////////////////////
    //Helper functions
    VectorXd doScaling( MatrixXd & data, const diffeoSearchOpts & opts){

        int dim = data.rows();
        int nPoints = data.cols();

        VectorXd scaling = VectorXd::Zero(dim);
        for (int i=0; i<dim; ++i){
            scaling(i) = sqrt(variance(data.row(i)));
        }
        switch(opts.thisScalingMethod){
            case each:{
                double meanScale;
                meanScale = scaling.mean();
                for (int i=0; i<dim; ++i){
                    if(scaling(i) < 1.e-2*meanScale){
                        //Do not scale if degenerated to avoid inf
                        scaling(i) = 1.;
                    }
                }
            }
            case minimal:{
                double minVar = scaling.minCoeff();//Use the minimal variance
                scaling = VectorXd::Constant(scaling.size(), minVar);
            }
            case maximal:{
                double maxVar = scaling.maxCoeff();//Use the minimal variance
                scaling = VectorXd::Constant(scaling.size(), maxVar);
            }
        }
        //Do the actual scaling
        data = data.cwiseQuotient( scaling.replicate(1, nPoints) );
        return scaling;
    }
    ///////////////////////////////////////////////
    void undoScaling(MatrixXd & data, const VectorXd & scaling){
        if (not (data.rows()==scaling.rows())){
            throw std::runtime_error("Data and scaling do not have the same row dimension");
        }
        data = data.cwiseProduct( scaling.replicate(1,data.cols()) );
    }
    ///////////////////////////////////////////////

    /*Functor as cost*/
    struct basicCost{
        public:
            MatrixXd const * source; //Check where to put const
            MatrixXd current;
            MatrixXd const * target; //Check where to put const
            int* idMax = nullptr;
            double const * regularisation = nullptr;
            double* division = nullptr;
            double*   coefMax = nullptr;
            SelfAdjointEigenSolver<MatrixXd> es;

            double operator()(const double & thisCoef) {
                //Calculate
                //Set before
                current = (*source);

                //MatrixXd current = *source; //Nahh
                //Get transformation
                //inline void iterativeFunction(const V1 & center, const V1 & target, M1 & pt, const double & divisionCoef, const double & coef)
                //iterativeFunction(VectorXd(current.col(*idMax)), VectorXd(target->col(*idMax)), current, *division, thisCoef);
                iterativeFunction(current.col(*idMax), target->col(*idMax), current, *division, thisCoef);
                //iterativeFunction(current.col(*idMax), (*target).col(*idMax), current, *division, thisCoef);
                //Get the cost; Matrix 2 norm -> largest singular value
                current = current - *target;
                //Is this the fastest way?
                //JacobiSVD<MatrixXd> svd(current*current.transpose());
                //double largestSingularValue = sqrt(svd.singularValues().maxCoeff());
                //SelfAdjointEigenSolver<MatrixXd> es;
                es.compute(current*current.transpose());
                //cout << target->col(500) << endl;
                double largestSingularValue = sqrt(es.eigenvalues().maxCoeff());


                //Regularise and return
                //cout << *regularisation << " " << *idMax << " " << *division << " " << thisCoef << " " << largestSingularValue/((double)current.cols())+(*regularisation)*thisCoef*thisCoef << endl;
                return largestSingularValue/((double)current.cols())+(*regularisation)*thisCoef*thisCoef;
            }
    };

    /*Function searching for the best exponential coefficient given source target id and options.
    Searches via golden section search and using a starting guess
    */
    template<typename aFunc>
    double goldenSectionSearch( aFunc & costFunc, const double & lowerLimit, const double & upperLimit, const double & conv = 1e-8){ //const int idMax, const M1 & current, const M1 & target, double const & coefUpper, const diffeoSearchOpts & opts, const double & lastCoef, const double & convCrit = 1e-3){
        double a = lowerLimit;
        double b = upperLimit;

        double goldenSection = (sqrt(5) + 1) / 2;

        double c = b - (b - a) / goldenSection;
        double d = a + (b - a) / goldenSection;


        while( abs(c - d) > conv ){

                if (costFunc(c)<costFunc(d)){
                    b=d;
                }else{
                    a=c;
                }
                c = b - (b - a) / goldenSection;
                d = a + (b - a) / goldenSection;
        }
        return (a+b)/2.;
    }

    /* Brute force */
    template<typename aFunc>
    double brutForceSearch( aFunc & costFunc, const double & lowerLimit, const double & upperLimit, const int nSteps = 1000){
        VectorXd searchPoints(nSteps);
        searchPoints.setLinSpaced(nSteps, lowerLimit, upperLimit);
        VectorXd searchValues(nSteps);
        int bestCoef;
        for (int i = 0; i<nSteps; ++i){
            searchValues(i) = costFunc( (double) searchPoints(i) );
            //cout << searchPoints(i) << " " << searchValues(i)  << endl;
        }
        //cout << searchValues << endl;
        double bestValue = searchValues.minCoeff(&bestCoef);
        //cout << bestValue << endl;
        //cout << bestCoef << endl;
        return searchPoints(bestCoef);
    }



    /*Default options
    */
    diffeoSearchOpts standardOpts;
    /*
    Function searching a diffeomorphism mapping the points in source onto the points of target.
    It is strongly recommended to normalize the data to have variance = 1 in each dimension.
    */
    template<typename M1>
    bool iterativeSearch( const M1 & source, const M1 & target, diffeoStruct & result, diffeoSearchOpts & opts = standardOpts ){

        const int dim = source.rows();
        const int nPoints = source.cols();

        //VectorXd divisionCoef;
        VectorXd safeCoef;

        //Check of options are consistent
        cout << opts.isUpToDate << endl;
        opts.doInterpolate();
        cout << opts.isUpToDate << endl;
        bool success = opts.getDivisionNSafe(result.divisionCoefs, safeCoef);
        if (not success){
            result = diffeoStruct();
            cout << "failed to retrieve coefs" << endl;
            return false;
        }

        //Initialize result struct
        result.numTrans = opts.numIterations;
        //result.centers = M1::Zero();
        //result.targets = M1::Zero();
        result.centers = MatrixXd::Zero(dim, result.numTrans);
        result.targets = MatrixXd::Zero(dim, result.numTrans);
        result.coefs = VectorXd::Zero(opts.numIterations);
        //result.divisionCoefs = divisionCoef;

        //begin the actual search
        M1 current = source;

        //helpers
        int numIt = opts.numIterations;
        const double maxCoef = opts.maxCoef;
        const double regularise = opts.regularise;
        const double convCrit = opts.convCrit*opts.convCrit;

        int idMax;
        double distMax;
        double coefUpper;

        //Get the struct used as functor
        basicCost thisCost;
        thisCost.source = &current;
        thisCost.target = &target;
        thisCost.regularisation = &regularise;
        thisCost.idMax = &idMax;
        thisCost.coefMax = &coefUpper;
        int i;
        for (i=0; i<numIt-1; ++i){
            //distMax = (target-current).array().square().colwise().sum().maxCoef(&idMax);
            distMax = (target-current).colwise().squaredNorm().maxCoeff(&idMax);

            //Check if converged                return largestSingularValue/((double)current.cols())+(*regularisation)*thisCoef*thisCoef;

            if (distMax < convCrit){
                result.numTrans = i;//i-1 iterations here plus final transformation
                result.centers.conservativeResize(dim, i); //#schlepil TBD verify resize
                result.targets.conservativeResize(dim, i);
                result.coefs.conservativeResize(i);
                result.divisionCoefs.conservativeResize(i);
                break;
            }

            //Get the largest admissible coef
            coefUpper = min(safeCoef(i)*result.divisionCoefs(i)/(exp(-1./2.)*sqrt(2.)*( sqrt(distMax) )), maxCoef);

            //Set the current divisionCoef
            thisCost.division = &result.divisionCoefs(i);
            //Search the best coef
            //the cost structure uses pointers so no need to update other values
            result.coefs(i) = goldenSectionSearch( thisCost, 0., coefUpper);
            //cout << result.coefs(i) << endl;
            //result.coefs(i) = brutForceSearch( thisCost, 0., coefUpper);
            //cout << result.coefs(i) << endl;

            //Save the rest
            result.centers.col(i) = current.col(idMax);
            result.targets.col(i) = target.col(idMax);
            //Update current
            iterativeFunction(result.centers.col(i), result.targets.col(i), current, result.divisionCoefs(i), result.coefs(i));

            //cout << result.divisionCoefs(i) << " " << result.coefs(i) << " " << *thisCost.division << " " << i << endl;
        }
        //Perform final transformation to match end-points
        idMax = nPoints-1;
        coefUpper = min(safeCoef(i)*1.*exp(-1./2.)/sqrt(2.)*( (target.col(idMax) - current.col(idMax)).norm() ), maxCoef);

        result.divisionCoefs(i) = 1.;
        thisCost.division = &result.divisionCoefs(i);
        result.coefs(i) = goldenSectionSearch( thisCost, 0., coefUpper);
        //result.coefs(i) = brutForceSearch( thisCost, 0., coefUpper);
        result.centers.col(i) = current.col(idMax);
        result.targets.col(i) = target.col(idMax);
        iterativeFunction(result.centers.col(i), result.targets.col(i), current, result.divisionCoefs(i), result.coefs(i));
        Leph::WriteMatrix("current", current, 16, "python");
        //Done
        return true;
    }

}

#endif // DIFFEO_SEARCH_HPP
