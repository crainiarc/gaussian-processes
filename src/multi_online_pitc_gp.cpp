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
    mOutputDimensions = obs.n_cols;
    mTrainingSet = data;
    mObservations = linearizeObservations(obs);
    mGlobalD = arma::Mat<double>();
    mGlobalE = arma::Mat<double>();
    
    if (mAutoLearn) {
        learn();
    }
}

auto MultiOutputOnlinePITCGP::addTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &obs) ->void {
    mOutputDimensions = obs.n_cols;
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
    arma::Mat<double> DSummary = arma::zeros<arma::Mat<double>>(mLatentVariables.n_rows, mLatentVariables.n_rows);
    arma::Mat<double> ESummary = arma::zeros<arma::Mat<double>>(mLatentVariables.n_rows, 1);
    auto N = mTrainingSet.n_rows / mOutputDimensions;
    
    // Compute slice summary
    for (auto q = 0; q < mOutputDimensions; ++q) {
        auto X_qt = mTrainingSet.rows(q * N, (q * N) + N);
        auto Y_qt = mObservations.rows(q * N, (q * N) + N);
        auto K_Xt_u = computeKfuSingular(X_qt, q);
        
        auto FX_qt = computeKffSingular(K_Xt_u, q, q);
        FX_qt -= K_Xt_u * arma::solve(mKuuCholesky.t(), arma::solve(mKuuCholesky, K_Xt_u.t())); // Add noise?
        
        auto FX_qtCholesky = arma::chol(FX_qt);
        auto v = arma::solve(FX_qtCholesky, K_Xt_u);
        DSummary += v.t() * v;
        ESummary += K_Xt_u.t() * arma::solve(FX_qtCholesky.t(), arma::solve(FX_qtCholesky, Y_qt));
    }
    
    // Assimilate
    mGlobalD += DSummary;
    mGlobalE += ESummary;
    
    // Clear the slice
    mTrainingSet.set_size(0, mTrainingSet.n_cols);
    mObservations.set_size(0, mOutputDimensions);
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
    
    mKuu = computeKuu(mLatentVariables);
    mKuuCholesky = arma::chol(mKuu);
}

auto MultiOutputOnlinePITCGP::linearizeObservations(const arma::Mat<double> &obs) -> arma::Col<double> {
    auto observations = arma::Col<double>();

    for (auto i = 0; i < obs.n_cols; ++i) {
        observations.insert_rows(observations.n_rows, obs.col(i));
    }
    
    return observations;
}

auto MultiOutputOnlinePITCGP::computeKff(const arma::Mat<double> &X) -> arma::Mat<double> {
    auto kff = arma::Mat<double>(X.n_rows, X.n_rows);
    auto N = X.n_rows / mOutputDimensions;
    
    for (auto q1 = 0; q1 < mOutputDimensions; ++q1) {
        for (auto q2 = 0; q2 < mOutputDimensions; ++q2) {
            for (auto n1 = 0; n1 < N; ++n1) {
                auto idx1 = (q1 * N) + n1;
                for (auto n2 = 0; n2 < n1; ++n2) {
                    auto idx2 = (q2 * N) + n2;
                    
                    kff(idx1, idx2) = ggXggKernCompute(X.row(idx1).t(), X.row(idx2).t(), q1 + 1, q2 + 1);
                    kff(idx2, idx1) = kff(idx1, idx2);
                }
            }
        }
        
        for (auto n1 = 0; n1 < N; ++n1) {
            auto idx = (q1 * N) + n1;
            kff(idx, idx) = whiteKernCompute(q1 + 1);
        }
    }
    
    return kff;
}

auto MultiOutputOnlinePITCGP::computeKffSingular(const arma::Mat<double> &X, int q1, int q2) -> arma::Mat<double> {
    auto kff = arma::Mat<double>(X.n_rows, X.n_rows);
    
    for (auto n1 = 0; n1 < X.n_rows; ++n1) {
        for (auto n2 = 0; n2 < n1; ++n2) {
            kff(n1, n2) = ggXggKernCompute(X.row(n1).t(), X.row(n2).t(), q1 + 1, q2 + 1);
            kff(n2, n1) = kff(n1, n2);
        }
    }
    
    if (q1 == q2) {
        for (auto n = 0; n < X.n_rows; ++n) {
            kff(n, n) = whiteKernCompute(q1 + 1);
        }
    }
    
    return kff;
}

auto MultiOutputOnlinePITCGP::computeKfu(const arma::Mat<double> &X) -> arma::Mat<double> {
    auto kfu = arma::Mat<double>(X.n_rows, mLatentVariables.n_rows);
    auto N = X.n_rows / mOutputDimensions;
    
    for (auto q = 0; q < mOutputDimensions; ++q) {
        for (auto n = 0; n < N; ++n) {
            for (auto i = 0; i < mLatentVariables.n_rows; ++i) {
                auto idx1 = (q*N) + n;
                kfu(idx1, i) = ggXgaussKernCompute(X.row(idx1).t(), mLatentVariables.row(i).t(), q + 1);
            }
        }
    }
    
    return kfu;
}

auto MultiOutputOnlinePITCGP::computeKfuSingular(const arma::Mat<double> &X, int q) -> arma::Mat<double> {
    auto kfu = arma::Mat<double>(X.n_rows, mLatentVariables.n_rows);
    
    for (auto n = 0; n < X.n_rows; ++n) {
        for (auto i = 0; i < mLatentVariables.n_rows; ++i) {
            kfu(n, i) = ggXgaussKernCompute(X.row(n).t(), mLatentVariables.row(i).t(), q + 1);
        }
    }
    
    return kfu;
}

auto MultiOutputOnlinePITCGP::computeKuu(const arma::Mat<double> &X) -> arma::Mat<double> {
    auto kuu = arma::Mat<double>(X.n_rows, X.n_rows);
    for (auto i = 0; i < X.n_rows; ++i) {
        for (auto j = 0; j < i; ++j) {
            kuu(i, j) = gaussKernCompute(X.row(i).t(), X.row(j).t());
            kuu(j, i) = kuu(i, j);
        }
        kuu(i, i) = gaussKernCompute(X.row(i).t(), X.row(i).t());
        kuu(i, i) += whiteKernCompute(0);
    }
    
    return kuu;
}

auto MultiOutputOnlinePITCGP::computeKtf(const arma::Mat<double> &X_star, const arma::Mat<double> &X) -> arma::Mat<double> {
    auto ktf = arma::Mat<double>(X_star.n_rows, X_star.n_rows);
    auto N_star = X_star.n_rows / mOutputDimensions;
    auto N = X.n_rows / mOutputDimensions;
    
    for (auto q1 = 0; q1 < mOutputDimensions; ++q1) {
        for (auto q2 = 0; q2 < mOutputDimensions; ++q2) {
            for (auto n1 = 0; n1 < N_star; ++n1) {
                for (auto n2 = 0; n2 < N; ++n2) {
                    auto idx1 = (q1 * N_star) + n1;
                    auto idx2 = (q2 * N) + n2;
                    ktf(idx1, idx2) = ggXggKernCompute(X_star.row(idx1).t(), X.row(idx2).t(), q1 + 1, q2 + 1);
                }
            }
        }
    }
    return ktf;
}

auto MultiOutputOnlinePITCGP::computeKDiag(int pos) -> double {
    return whiteKernCompute(pos) + ggDiagKernCompute(pos);
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