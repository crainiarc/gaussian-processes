//
//  gp.cpp
//  GaussianProcess
//
//  Created by Keng Kiat Lim on 1/9/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#include "gp.h"

GaussianProcess::GaussianProcess(kernel_func_t kernelFunc, double noiseVar) :
    mKernelFunction(kernelFunc), mNoiseVariance(noiseVar) {
        mAutoLearn = true;
}

auto GaussianProcess::setTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &observations) -> void {
    mTrainingSet = data;
    mObservations = observations;
}

auto GaussianProcess::addTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &observations) -> void {
}

auto GaussianProcess::learn() -> void {
    // Compute the cholesky decomposition of the training set
    mTrainingCovariances = covarianceMatrix(mTrainingSet, mTrainingSet);
    mTrainingCovariances += mNoiseVariance * arma::eye<arma::Mat<double>>(mTrainingCovariances.n_rows,
                                                                          mTrainingCovariances.n_cols);
    mCholesky = arma::chol(mTrainingCovariances);
    
    // Compute the alpha as detailed in Algorithm 2.1 of GPML
    auto y = arma::solve(mCholesky, mObservations);
    mAlpha = arma::solve(mCholesky.t(), y);
}

auto predict(const arma::Mat<double> &testData) -> std::tuple<arma::Mat<double>, arma::Mat<double>> {
    return std::make_tuple(arma::Mat<double>(), arma::Mat<double>());
}

auto predictMean(const arma::Mat<double> &testData) -> arma::Mat<double> {
    return arma::Mat<double>();
}

auto GaussianProcess::covarianceMatrix(const arma::Mat<double> X, const arma::Mat<double> Y) -> arma::Mat<double> {
    // TODO: train hyperparameters
    
    arma::Mat<double> covMat(X.n_cols, Y.n_cols);
    for (unsigned int i = 0; i < X.n_cols; ++i) {
        for (unsigned int j = 0; j < X.n_cols; ++j) {
            covMat(i, j) = mKernelFunction(X.col(i), X.col(j));
        }
    }
    return arma::Mat<double>();
}