//
//  sparse_pitc.cpp
//  GaussianProcess
//
//  Created by Keng Kiat Lim on 18/1/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#include "sparse_pitc.h"

PITCSparseGP::PITCSparseGP(kernel_func_t kernelFunc, double noiseVar, int blockSize) :
    GaussianProcess(kernelFunc, noiseVar), mBlockSize(blockSize) {
}

PITCSparseGP::PITCSparseGP(kernel_func_t kernelFunc, double noiseVar, int blockSize, bool autoLearn) :
    GaussianProcess(kernelFunc, noiseVar, autoLearn), mBlockSize(blockSize) {
}

auto PITCSparseGP::setInducingInputs(const arma::Mat<double> &inducingInputs) -> void {
    mInducingInputs = inducingInputs;
    mInducingCovariancesInverse = covarianceMatrix(mInducingInputs, mInducingInputs).i();
    
    if (mAutoLearn) {
        learn();
    }
}

auto PITCSparseGP::learn() -> void {
}

auto PITCSparseGP::predict(const arma::Mat<double> &testData) -> std::tuple<arma::Mat<double>, arma::Mat<double>> {
    return arma::Mat<double>();
}

auto PITCSparseGP::predictMean(const arma::Mat<double> &testData) -> arma::Mat<double>  {
    return arma::Mat<double>();
}

auto PITCSparseGP::predictVariance(const arma::Mat<double> &testData) -> arma::Mat<double> {
    return arma::Mat<double>();

auto PITCSparseGP::blockDiagonal(const arma::Mat<double> &mat, const int blockSize) -> arma::Mat<double> {
    if (mat.n_cols != mat.n_rows) {
        // Not a square matrix
        // TODO: Throw error
        return arma::Mat<double>();
    }
    if (mat.n_cols % blockSize != 0) {
        // Matrix cannot be blocked evenly
        // TODO: Throw error
        return arma::Mat<double>();
    }
    
    arma::Mat<double> bMat = arma::Mat<double>(mat.n_rows, mat.n_cols);
    for (unsigned int i = 0; i < mat.n_cols; ++i) {
        auto lastRow = i + blockSize - 1;
        bMat.submat(i, i, lastRow, lastRow) = mat.submat(i, i, lastRow, lastRow);
    }
    return bMat;
}

auto PITCSparseGP::computeQ(const arma::Mat<double> &a, const arma::Mat<double> &b) -> arma::Mat<double> {
    auto K_au = covarianceMatrix(a, mInducingInputs);
    
    if (&a == &b) {
        return K_au * mInducingCovariancesInverse * K_au.i();
    } else {
        auto K_ub = covarianceMatrix(mInducingInputs, b);
        return K_au * mInducingCovariancesInverse * K_ub;
    }
}

auto PITCSparseGP::computeBigSigma(const arma::Mat<double> K_uu, const arma::Mat<double> K_fu, const arma::Mat<double> K_ufLambdaInverse) -> arma::Mat<double> {
    return (K_uu + (K_ufLambdaInverse * K_fu)).i();
}