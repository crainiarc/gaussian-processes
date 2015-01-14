//
//  gp.h
//  GaussianProcess
//
//  Created by Keng Kiat Lim on 1/9/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#ifndef __GaussianProcess__gp__
#define __GaussianProcess__gp__

#include <armadillo>

typedef std::function<double(arma::Col<double>, arma::Col<double>)> kernel_func_t;

class GaussianProcess {
    friend class GaussianProcessTest;
    
public:
    GaussianProcess(kernel_func_t kernelFunc, double noiseVar);
    GaussianProcess(kernel_func_t kernelFucc, double noiseVar, bool autoLearn);
    
    auto setTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &observations) -> void;
    auto addTrainingSet(const arma::Mat<double> &data, const arma::Mat<double> &observations) -> void;
    auto setAutoLearn(bool autoLearn) -> void;
    auto learn() -> void;
    auto predict(const arma::Mat<double> &testData) -> std::tuple<arma::Mat<double>, arma::Mat<double>>;
    auto predictMean(const arma::Mat<double> &testData) -> arma::Mat<double>;
    auto predictVariance(const arma::Mat<double> &testData) -> arma::Mat<double>;
    
protected:
    bool mAutoLearn;
    kernel_func_t mKernelFunction;
    double mNoiseVariance;
    arma::Mat<double> mTrainingSet;
    arma::Mat<double> mObservations;
    
    arma::Mat<double> mTrainingCovariances;
    arma::Mat<double> mCholesky;
    arma::Col<double> mAlpha;
    
    virtual auto covarianceMatrix(const arma::Mat<double> X, const arma::Mat<double> Y) -> arma::Mat<double>;
};

#endif /* defined(__GaussianProcess__gp__) */
