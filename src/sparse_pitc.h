//
//  sparse_pitc.h
//  GaussianProcess
//
//  Created by Keng Kiat Lim on 18/1/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#ifndef __GaussianProcess__sparse_pitc__
#define __GaussianProcess__sparse_pitc__

#include <armadillo>
#include "gp.h"

class PITCSparseGP : public GaussianProcess {
    friend class PITCSparseGPTest;
    
public:
    PITCSparseGP(kernel_func_t kernelFunc, double noiseVar, int blockSize);
    PITCSparseGP(kernel_func_t kernelFuct, double noiseVar, int blockSize, bool autoLearn);
    auto setInducingInputs(const arma::Mat<double> &inducingInputs) -> void;
    
    virtual auto learn() -> void override;
    virtual auto predict(const arma::Mat<double> &testData) -> std::tuple<arma::Mat<double>, arma::Mat<double>> override;
    virtual auto predictMean(const arma::Mat<double> &testData) -> arma::Mat<double> override;
    virtual auto predictVariance(const arma::Mat<double> &testData) -> arma::Mat<double> override;
    
protected:
    int mBlockSize;
    arma::Mat<double> mInducingInputs;
    arma::Mat<double> mInducingCovariancesInverse;
    
    auto blockDiagonal(const arma::Mat<double> &mat, const int blockSize) -> arma::Mat<double>;
    auto computeQ(const arma::Mat<double> &a, const arma::Mat<double> &b) -> arma::Mat<double>;
    
};

#endif /* defined(__GaussianProcess__sparse_pitc__) */
