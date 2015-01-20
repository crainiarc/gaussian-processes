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
    
protected:
    int mBlockSize;
    arma::Mat<double> mInducingInputs;
    arma::Mat<double> mInducingCovariances;
};

#endif /* defined(__GaussianProcess__sparse_pitc__) */
