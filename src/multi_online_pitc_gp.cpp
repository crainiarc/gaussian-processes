//
//  multi_online_pitc_gp.cpp
//  GaussianProcess
//
//  Created by Keng Kiat Lim on 8/3/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#include "multi_online_pitc_gp.h"

MultiOutputOnlinePITCGP::MultiOutputOnlinePITCGP(int blockSize, const arma::Mat<double> &latentVars, MultiOutputHyperparameters hypers) :
    mBlockSize(blockSize), mLatentVariables(latentVars), mHypers(hypers)
{
    mAutoLearn = true;
    initHyperparameters();
}

MultiOutputOnlinePITCGP::MultiOutputOnlinePITCGP(int blockSize, const arma::Mat<double> &latentVars, MultiOutputHyperparameters hypers, bool autoLearn) :
    mBlockSize(blockSize), mLatentVariables(latentVars), mHypers(hypers), mAutoLearn(autoLearn)
{
    initHyperparameters();
}

auto MultiOutputOnlinePITCGP::setTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &obs) -> void {
    mTrainingSet = data;
    mObservations = linearizeObservations(obs);
    mGlobalD = arma::Mat<double>();
    mGlobalE = arma::Mat<double>();
    
    if (mAutoLearn) {
        learn();
    }
}

auto MultiOutputOnlinePITCGP::addTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &obs) ->void {
    mTrainingSet.insert_cols(mTrainingSet.n_cols, data);
    mObservations.insert_rows(mObservations.n_rows, linearizeObservations(obs));
    
    if (mAutoLearn) {
        learn();
    }
}

auto MultiOutputOnlinePITCGP::setHyperParameters(MultiOutputHyperparameters hypers) -> void {
    mHypers = hypers;
    initHyperparameters();
}

auto MultiOutputOnlinePITCGP::setAutoLearn(bool autoLearn) -> void {
    mAutoLearn = autoLearn;
}

auto MultiOutputOnlinePITCGP::learn() -> void {
}

auto MultiOutputOnlinePITCGP::predict(const arma::Mat<double> &testData) -> std::tuple<arma::Mat<double>, arma::Mat<double>> {
    return std::make_tuple(arma::Mat<double>(), arma::Mat<double>());
}

auto MultiOutputOnlinePITCGP::predictMean(const arma::Mat<double> &testData) -> arma::Mat<double> {
    return arma::Mat<double>();
}

auto MultiOutputOnlinePITCGP::predictVariance(const arma::Mat<double> &testData) -> arma::Mat<double> {
    return arma::Mat<double>();
}

auto MultiOutputOnlinePITCGP::initHyperparameters() -> void {
    mHypers.precisionYsInv = 1.0 / mHypers.precisionYs;
    mHypers.precisionUsInv = 1.0 / mHypers.precisionUs;
}

auto MultiOutputOnlinePITCGP::linearizeObservations(const arma::Mat<double> &obs) -> arma::Col<double> {
    auto observations = arma::Col<double>();

    for (auto i = 0; i < obs.n_cols; ++i) {
        observations.insert_rows(observations.n_rows, obs.col(i));
    }
    
    return observations;
}

auto MultiOutputOnlinePITCGP::computeKff(const arma::Mat<double> &X, int q) -> arma::Mat<double> {
    return arma::Mat<double>();
}

auto MultiOutputOnlinePITCGP::computeKfu(const arma::Mat<double> &X, int q) -> arma::Mat<double> {
    return arma::Mat<double>();
}

auto MultiOutputOnlinePITCGP::computeKuu(const arma::Mat<double> &X) -> arma::Mat<double> {
    return arma::Mat<double>();
}

auto MultiOutputOnlinePITCGP::computeKtf(const arma::Mat<double> &X_star, const arma::Mat<double> &X, int q) -> arma::Mat<double> {
    return arma::Mat<double>();
}

auto MultiOutputOnlinePITCGP::computekDiag(int pos) -> double {
    return 0.0;
}

auto MultiOutputOnlinePITCGP::gaussKernCompute(const arma::Col<double> &x, const arma::Col<double> &y) -> double {
    return 0.0;
}

auto MultiOutputOnlinePITCGP::ggXggKernCompute(const arma::Col<double> &x, const arma::Col<double> &y, int posX, int posY) -> double {
    return 0.0;
}

auto MultiOutputOnlinePITCGP::ggXgaussKernCompute(const arma::Col<double> &x, const arma::Col<double> &y, int pos) -> double {
    return 0.0;
}

auto MultiOutputOnlinePITCGP::whiteKernCompute(int pos) -> double {
    return 0.0;
}

auto MultiOutputOnlinePITCGP::ggDiagKernCompute(int pos) -> double {
    return 0.0;
}