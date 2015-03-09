//
//  multi_online_pitc_gp.h
//  GaussianProcess
//
//  Created by Keng Kiat Lim on 8/3/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#ifndef __GaussianProcess__multi_online_pitc_gp__
#define __GaussianProcess__multi_online_pitc_gp__

#include <armadillo>

struct MultiOutputHyperparameters {
    arma::Col<double> biases;
    arma::Col<double> scales;
    arma::Col<double> betas;
    
    arma::Col<double> variances;
    arma::Col<double> sigma2Ys;
    arma::Col<double> sigma2Us;
    
    arma::Mat<double> precisionYs;
    arma::Mat<double> precisionUs;
};


class MultiOutputOnlinePITCGP {
    friend class MultiOutputOnlinePITCGPTest;
    
public:
    MultiOutputOnlinePITCGP(int blockSize, arma::Mat<double> latentVars, MultiOutputHyperparameters hypers);
    MultiOutputOnlinePITCGP(int blockSize, arma::Mat<double> latentVars, MultiOutputHyperparameters hypers, bool autoLearn);
    
    auto setTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &obs) -> void;
    auto addTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &obs) -> void;
    auto setHyperParameters(MultiOutputHyperparameters hypers) -> void;
    auto setAutoLearn(bool autoLearn) -> void;
    
    auto learn() -> void;
    auto predict(const arma::Mat<double> &testData) -> std::tuple<arma::Mat<double>, arma::Mat<double>>;
    auto predictMean(const arma::Mat<double> &testData) -> arma::Mat<double>;
    auto predictVariance(const arma::Mat<double> &testData) -> arma::Mat<double>;
    
protected:
    int mBlockSize;
    bool mAutoLearn;
    
    arma::Mat<double> mTrainingSet;
    arma::Mat<double> mLatentVariables;
    arma::Col<double> mObservations;
    MultiOutputHyperparameters mHyperparameters;
    
    arma::Mat<double> mGlobalD;
    arma::Mat<double> mGlobalE;
    
    auto initHyperparameters() -> void;
    auto linearizeObservations(const arma::Mat<double> &obs) -> arma::Col<double>;
    auto computeKff(arma::Mat<double> X, int q) -> arma::Mat<double>;
    auto computeKfu(arma::Mat<double> X, int q) -> arma::Mat<double>;
    auto computeKuu(arma::Mat<double> X) -> arma::Mat<double>;
    auto computeKtf(arma::Mat<double> X_star, arma::Mat<double> X, int q) -> arma::Mat<double>;
};

#endif /* defined(__GaussianProcess__multi_online_pitc_gp__) */
