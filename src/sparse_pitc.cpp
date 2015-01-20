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