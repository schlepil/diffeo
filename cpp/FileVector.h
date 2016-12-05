#ifndef LEPH_FILEVECTOR_HPP
#define LEPH_FILEVECTOR_HPP

#include <string>
#include <Eigen/Core>

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace Schlepil {

void WriteVector(
    const std::string& filename,
    const Eigen::VectorXd& vect,
	const int prec=15,
	const std::string mode="cpp")
{
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        throw std::runtime_error(
            "WriteVector unable to open file: "
            + filename);
    }
    if (mode.compare("cpp") == 0){
        file << vect.size();
        for (size_t i=0;i<(size_t)vect.size();i++) {
            file << " " << std::setprecision(prec) << vect(i);
        }
    }else if(mode.compare("python") == 0){
        for (size_t i=0;i<(size_t)vect.size();i++) {
            file << " " << std::setprecision(prec) << vect(i);
        }
    }else{
        throw std::runtime_error( "Mode needs to be either cpp or python (which is also readable by matlab" );
    }

    file << std::endl;
    file.close();
}

void WriteMatrix(
    const std::string& filename,
    const Eigen::MatrixXd & mat,
    const int prec=15,
	const std::string mode="cpp")
{
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        throw std::runtime_error(
            "WriteMatrix unable to open file: "
            + filename);
    }
    if (mode.compare("cpp") == 0){
        file << mat.rows() << " ";
        file << mat.cols() << " ";

        for (size_t j=0;j<(size_t)mat.cols();j++){
            for (size_t i=0;i<(size_t)mat.rows();i++) {
                file << " " << std::setprecision(prec) << mat(i,j);
            }
        }
    }else if(mode.compare("python") == 0){
        std::stringstream thisRowStream;
        for(size_t i=0; i<(size_t)mat.rows(); ++i){
            thisRowStream.str(std::string());
            for (size_t j=0;j<(size_t)mat.cols();++j) {
                thisRowStream << std::setprecision(prec) << mat(i,j) << " "; //space delimited
            }
            file << thisRowStream.str() << std::endl;
        }
    }else{
        throw std::runtime_error( "Mode needs to be either cpp or python (which is also readable by matlab" );
    }

    file << std::endl;
    file.close();
}


Eigen::VectorXd ReadVector(
    const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(
            "ReadVector unable to open file: "
            + filename);
    }
    double sizeOfVec;
    file >>  sizeOfVec;
    if (((size_t) sizeOfVec)<1){
        throw std::runtime_error("Vector must contain at least one element");
    }
    Eigen::VectorXd vect( (size_t) sizeOfVec ) ;
    for (size_t i=0;i<(size_t) sizeOfVec;i++) {
        double value;
        file >> value;
        vect(i) = value;
    }
    file.close();
    return vect;
}

Eigen::MatrixXd ReadMatrix(
    const std::string& filename )
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(
            "ReadMatrix unable to open file: "
            + filename);
    }
    double rows, cols;
    file >> rows;
    file >> cols;
    if (((size_t) rows)<1){
        throw std::runtime_error("Matrix must have at least one row");
    }
    if (((size_t) cols)<1){
        throw std::runtime_error("Matrix must have at least one column");
    }

    Eigen::MatrixXd mat((size_t)rows, (size_t) cols);
    //column major order
    double value;
    for (size_t j=0;j<(size_t)cols;j++){
        for (size_t i=0;i<(size_t)rows;i++) {
			//This is probably not very efficient...
            file >> value;
            mat(i,j) = value;
        }
    }
    file.close();
    return mat;
}


}
#endif

