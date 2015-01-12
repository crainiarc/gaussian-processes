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
    
    if (mAutoLearn) {
        learn();
    }
}

auto GaussianProcess::addTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &observations) -> void {
    if (mAutoLearn) {
        learn();
    }
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

auto GaussianProcess::predict(const arma::Mat<double> &testData) -> std::tuple<arma::Mat<double>, arma::Mat<double>> {
    auto trainingTestCovariance = covarianceMatrix(mTrainingSet, testData);
    auto testCovariance = covarianceMatrix(testData, testData);
    auto v = arma::solve(mCholesky, trainingTestCovariance);
    
    auto mean = trainingTestCovariance.t() * mAlpha;
    auto variance = testCovariance - (v.t() * v);
    
    return std::make_tuple(mean, variance);
}

auto GaussianProcess::predictMean(const arma::Mat<double> &testData) -> arma::Mat<double> {
    auto trainingTestCovariance = covarianceMatrix(mTrainingSet, testData);
    return trainingTestCovariance.t() * mAlpha;
}

auto GaussianProcess::predictVariance(const arma::Mat<double> &testData) -> arma::Mat<double> {
    auto trainingTestCovariance = covarianceMatrix(mTrainingSet, testData);
    auto testCovariance = covarianceMatrix(testData, testData);
    auto v = arma::solve(mCholesky, trainingTestCovariance);
    
    return testCovariance - (v.t() * v);
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