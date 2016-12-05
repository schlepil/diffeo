#ifndef THINGSTHATSHOULDEXIST_HPP
#define THINGSTHATSHOULDEXIST_HPP

#include <vector>
#include <Eigen/Core>
#include<numeric>


using namespace std;
using namespace Eigen;



namespace thingsThatShouldExist{

inline double variance(const VectorXd & dataIn){
    return ((dataIn - dataIn.mean()*VectorXd::Ones(dataIn.size()))).array().square().sum()/dataIn.size();
}

//Sorting of vectors
//The vector itself will be modified; A vector containing the index is returned
template <typename T>
vector<size_t> doSort(vector<T> &v) {

    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0); //this fills the vector

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    //Get a sorted version of the vector
    vector<T> reOrdered(v.size());

    for (size_t i=0; i<idx.size(); ++i){
        reOrdered[i] = v[idx[i]];
    }
    v.swap(reOrdered);
    return idx;
}

//Sort Eigenvectors
template<typename dType>
Matrix<int, -1, 1> doSort( Matrix<dType, -1, 1> &vIn ){

    // initialize original index locations
    vector<size_t> idx(vIn.size());
    iota(idx.begin(), idx.end(), 0); //this fills the vector

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&vIn](size_t i1, size_t i2) {return vIn(i1) < vIn(i2);});

    //Get a sorted version of the vector
    Matrix<dType, -1, 1> reOrdered(vIn.size());
    Matrix<int, -1, 1> idxEigen(vIn.size());

    for (size_t i=0; i<idx.size(); ++i){
        reOrdered(i) = vIn(idx[i]);
        idxEigen(i) = idx[i];
    }
    vIn = reOrdered;

    return idxEigen;
}


}

#endif // THINGSTHATSHOULDEXIST_HPP
