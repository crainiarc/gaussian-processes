//
//  multi_online_pitc_gp.cpp
//  GaussianProcess
//
//  Created by Keng Kiat Lim on 8/3/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#include "multi_online_pitc_gp.h"

MultiOutputOnlinePITCGP::MultiOutputOnlinePITCGP(int blockSize, const arma::Mat<double> &latentVars, MultiOutputHyperparameters hypers) :
    mBlockSize(blockSize), mLatentVariables(latentVars), mHyperparameters(hypers)
{
    autoLearn = true;
    initHyperparameters();
}

MultiOutputOnlinePITCGP::MultiOutputOnlinePITCGP(int blockSize, const arma::Mat<double> &latentVars, MultiOutputHyperparameters hypers, bool autoLearn) :
    mBlockSize(blockSize), mLatentVariables(latentVars), mHyperparameters(hypers), mAutoLearn(autoLearn)
{
    initHyperparameters();
}

auto MultiOutputOnlinePITCGP::setTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &obs) {
    mTrainingSet = data;
    mObservations = linearizeObservations(obs);
    mGlobalD = arma::Mat<double>;
    mGlobalE = arma::Mat<double>;
    
    if (mAutoLearn) {
        learn();
    }
}

auto MultiOutputOnlinePITCGP::addTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &obs) {
    mTrainingSet.insert_cols(mTrainingSet.n_cols, data);
    mObservations.insert_rows(mObservations.n_rows, linearizeObservations(obs));
    
    if (mAutoLearn) {
        learn();
    }
}

auto MultiOutputOnlinePITCGP::setHyperParameters(MultiOutputHyperparameters hypers) {
    mHyperparameters = hypers;
    initHyperparameters();
}

auto MultiOutputOnlinePITCGP::setAutoLearn(bool autoLearn) {
    mAutoLearn = autoLearn;
}

auto MultiOutputOnlinePITCGP::learn() {
}

auto MultiOutputOnlinePITCGP::predict(const arma::Mat<double> &testData) -> std::tuple<arma::Mat<double>, arma::Mat<double>> {
    return std::make_tuple(arma::Mat<double>, arma::Mat<double>);
}

auto MultiOutputOnlinePITCGP::predictMean(const arma::Mat<double> &testData) -> arma::Mat<double> {
    return arma::Mat<double>;
}

auto MultiOutputOnlinePITCGP::predictVariance(const arma::Mat<double> &testData) -> arma::Mat<double> {
    return arma::Mat<double>;
}

auto MultiOutputOnlinePITCGP::initHyperparameters() {
}

auto linearizeObservations(const arma::Mat<double> &obs) -> arma::Col<double> {
}

auto MultiOutputOnlinePITCGP::computeKff(const arma::Mat<double> &X, int q) -> arma::Mat<double> {
    return arma::Mat<double>;
}

auto MultiOutputOnlinePITCGP::computeKfu(const arma::Mat<double> &X, int q) -> arma::Mat<double> {
    return arma::Mat<double>;
}

auto MultiOutputOnlinePITCGP::computeKuu(const arma::Mat<double> &X) -> arma::Mat<double> {
    return arma::Mat<double>;
}

auto MultiOutputOnlinePITCGP::computeKtf(const arma::Mat<double> &X_star, const arma::Mat<double> &X, int q) -> arma::Mat<double> {
    return arma::Mat<double>;
}