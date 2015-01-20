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
    mInducingCovariances = covarianceMatrix(mInducingInputs, mInducingInputs);
    
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