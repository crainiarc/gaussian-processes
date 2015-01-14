//
//  gp_test.h
//  GaussianProcess
//
//  Created by Keng Kiat Lim on 1/13/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#include <armadillo>
#include "gtest/gtest.h"
#include "gp.h"


auto squaredExponential = [](arma::Col<double> x, arma::Col<double> y) {
    auto norm = arma::norm(x - y, 2);
    return (double) exp(-(norm*norm) / 2.0);
};

class GaussianProcessTest : public ::testing::Test {
protected:
    GaussianProcessTest() :
        gp1(GaussianProcess(squaredExponential, 0.0, false)) {
    }
    
    auto SetUp() -> void {
    }
    
    auto getMTrainingSet(GaussianProcess gp) -> arma::Mat<double> {
        return gp.mTrainingSet;
    }
    
    auto getMObservations(GaussianProcess gp) -> arma::Mat<double> {
        return gp.mObservations;
    }
    
    GaussianProcess gp1;
};

TEST_F(GaussianProcessTest, TestSetTrainingSetNoAutoLearn) {
    auto nRows = 3, nCols = 3;
    arma::Mat<double> trainingSet = arma::randu<arma::Mat<double>>(nRows, nCols);
    arma::Mat<double> observations = arma::randu<arma::Mat<double>>(nRows, 1);
    gp1.setTrainingSet(trainingSet, observations);
    
    // Test that training set is equal
    for (auto i = 0; i < nRows; ++i) {
        for (auto j = 0; j < nCols; ++j) {
            ASSERT_EQ(trainingSet(i, j), getMTrainingSet(gp1)(i, j));
        }
    }
    
    // Test that observations are equal
    for (auto i = 0; i < nRows; ++i) {
        ASSERT_EQ(observations(i, 0), getMObservations(gp1)(i, 0));
    }
    
    // Test the the matrices are copied over to the object
    trainingSet(0, 0) = 1.0;
    observations(0, 0) = 1.0;
    ASSERT_NE(1.0, getMTrainingSet(gp1)(0, 0));
    ASSERT_NE(1.0, getMObservations(gp1)(0, 0));
}

TEST_F(GaussianProcessTest, TestAddTrainingSetNoAutoLearn) {
    
}

TEST_F(GaussianProcessTest, TestCovarianceMatrix) {
    
}