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

auto MultiOutputOnlinePITCGP::computeKDiag(int pos) -> double {
    return 0.0;
}

auto MultiOutputOnlinePITCGP::gaussKernCompute(const arma::Col<double> &x, const arma::Col<double> &y) -> double {
    auto result = arma::sum((arma::pow(x - y, 2) % mHypers.precisionUs.col(0)));
    return mHypers.sigma2Us(0) * exp(-0.5 * result);
}

auto MultiOutputOnlinePITCGP::ggXggKernCompute(const arma::Col<double> &x, const arma::Col<double> &y, int posX, int posY) -> double {
    auto detBkInv = arma::prod(mHypers.precisionUsInv.col(posX));
    auto preLInv = mHypers.precisionYsInv.col(posX) + mHypers.precisionYsInv.col(posY) + mHypers.precisionUsInv(posX);
    auto lDet = arma::prod(preLInv);
    auto lInv = 1.0 / preLInv;
    
    auto result = exp(-0.5 * arma::sum((arma::pow(x - y, 2) % lInv)));
    result *= mHypers.sigma2Ys(posX) * mHypers.sigma2Ys(posY) * mHypers.sigma2Us(posX);
    return result * sqrt(detBkInv / lDet);
}

auto MultiOutputOnlinePITCGP::ggXgaussKernCompute(const arma::Col<double> &x, const arma::Col<double> &y, int pos) -> double {
    auto detBkInv = arma::prod(mHypers.precisionUsInv.col(0));
    auto preLInv = mHypers.precisionYsInv.col(pos) + mHypers.precisionUsInv.col(0);
    auto lDet = arma::prod(preLInv);
    auto lInv = 1.0 / preLInv;
    
    auto result = exp(-0.5 * arma::sum((arma::prod(x - y, 2) % lInv)));
    result *= mHypers.sigma2Ys(pos) * mHypers.sigma2Us(0) * sqrt(detBkInv / lDet);
    return result;
}

auto MultiOutputOnlinePITCGP::whiteKernCompute(int pos) -> double {
    return mHypers.variances(pos);
}

auto MultiOutputOnlinePITCGP::ggDiagKernCompute(int pos) -> double {
    auto detBkInv = arma::prod(mHypers.precisionUsInv.col(pos));
    auto lDet = arma::prod((2.0 * mHypers.precisionYsInv.col(pos)) + mHypers.precisionUsInv.col(pos));
    auto result = mHypers.sigma2Ys(pos) * mHypers.sigma2Ys(pos) * mHypers.sigma2Us(pos);
    return result * sqrt(detBkInv / lDet);
}