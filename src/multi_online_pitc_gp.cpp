//
//  multi_online_pitc_gp.cpp
//  GaussianProcess
//
//  Created by Keng Kiat Lim on 8/3/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#include "multi_online_pitc_gp.h"

MultiOutputOnlinePITCGP::MultiOutputOnlinePITCGP(int blockSize, arma::Mat<double> latentVars, MultiOutputHyperparameters hypers) :
    mBlockSize(blockSize), mLatentVariables(latentVars), mHyperparameters(hypers)
{
    autoLearn = true;
}

MultiOutputOnlinePITCGP::MultiOutputOnlinePITCGP(int blockSize, arma::Mat<double> latentVars, MultiOutputHyperparameters hypers, bool autoLearn) :
    mBlockSize(blockSize), mLatentVariables(latentVars), mHyperparameters(hypers), mAutoLearn(autoLearn)
{
}

auto MultiOutputOnlinePITCGP::setTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &obs) {
}

auto MultiOutputOnlinePITCGP::addTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &obs) {
}

auto MultiOutputOnlinePITCGP::setHyperParameters(MultiOutputHyperparameters hypers) {
}

auto MultiOutputOnlinePITCGP::setAutoLearn(bool autoLearn) {
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

auto MultiOutputOnlinePITCGP::computeKff(arma::Mat<double> X, int q) -> arma::Mat<double> {
    return arma::Mat<double>;
}

auto MultiOutputOnlinePITCGP::computeKfu(arma::Mat<double> X, int q) -> arma::Mat<double> {
    return arma::Mat<double>;
}

auto MultiOutputOnlinePITCGP::computeKuu(arma::Mat<double> X) -> arma::Mat<double> {
    return arma::Mat<double>;
}

auto MultiOutputOnlinePITCGP::computeKtf(arma::Mat<double> X_star, arma::Mat<double> X, int q) -> arma::Mat<double> {
    return arma::Mat<double>;
}