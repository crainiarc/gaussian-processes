//
//  sparse_pitc_test.cpp
//  GaussianProcess
//
//  Created by Keng Kiat Lim on 21/1/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#include <armadillo>
#include <stdexcept>
#include "gtest/gtest.h"
#include "sparse_pitc.h"

auto squaredExponential = [](arma::Mat<double> x, arma::Mat<double> y) {
    auto norm = arma::norm(x - y, 2);
    return (double) exp(-(norm*norm) / 2.0);
};

class PITCSparseGPTest : public ::testing::Test {
protected:
    PITCSparseGPTest() :
        gp1(PITCSparseGP(squaredExponential, 0.0, 1, false)),
        matrix1(arma::randu<arma::Mat<double>>(12, 12)) {
    }
    
    auto blockDiagonalTest(PITCSparseGP gp, arma::Mat<double> mat, int blockSize) -> arma::Mat<double> {
        return gp.blockDiagonal(mat, blockSize);
    }
    
    arma::Mat<double> matrix1;
    PITCSparseGP gp1;
};

// Tests blockDiagonal for the case where block size = 1 (a full diagonal matrix)
TEST_F(PITCSparseGPTest, TestDiagonalMatrix) {
    auto diagMat = blockDiagonalTest(gp1, matrix1, 1);
    for (unsigned int i = 0; i < matrix1.n_rows; ++i) {
        for (unsigned int j = 0; j < matrix1.n_cols; ++j) {
            if (i == j) {
                ASSERT_DOUBLE_EQ(matrix1(i, j), diagMat(i, j));
            } else {
                ASSERT_DOUBLE_EQ(0.0, diagMat(i, j));
            }
        }
    }
}

// Tests that blockGiagonal generates an actual block diagonal matrix
TEST_F(PITCSparseGPTest, TestBlockDiagonalMatrix) {
    arma::Mat<double> testMat = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
    testMat.reshape(6, 6);
    
    arma::Mat<double> expectedMat = {1, 2, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 15, 16, 0, 0,
        0, 0, 21, 22, 0, 0, 0, 0, 0, 0, 29, 30, 0, 0, 0, 0, 35, 36};
    expectedMat.reshape(6, 6);
    auto blockDiagMat1 = blockDiagonalTest(gp1, testMat, 2);
    for (unsigned int i = 0; i < testMat.n_rows; ++i) {
        for (unsigned int j = 0; j < testMat.n_cols; ++j) {
            ASSERT_FLOAT_EQ(expectedMat(i, j), blockDiagMat1(i, j));
        }
    }
    
    expectedMat = {1, 2, 3, 0, 0, 0, 7, 8, 9, 0, 0, 0, 13, 14, 15, 0, 0, 0,
        0, 0, 0, 22, 23, 24, 0, 0, 0, 28, 29, 30, 0, 0, 0, 34, 35, 36};
    expectedMat.reshape(6, 6);
    auto blockDiagMat2 = blockDiagonalTest(gp1, testMat, 3);
    for (unsigned int i = 0; i < testMat.n_rows; ++i) {
        for (unsigned int j = 0; j < testMat.n_cols; ++j) {
            ASSERT_FLOAT_EQ(expectedMat(i, j), blockDiagMat2(i, j));
        }
    }
}

TEST_F(PITCSparseGPTest, TestBlockDiagonalMatrixNonSquareMatrix) {
    arma::Mat<double> testMat = arma::randu<arma::Mat<double>>(3, 5);
    ASSERT_THROW(auto mat = blockDiagonalTest(gp1, testMat, 1), std::logic_error);
}

TEST_F(PITCSparseGPTest, TestBlockDiagonalMatrixNonBLockableMatrix) {
    ASSERT_THROW(auto mat = blockDiagonalTest(gp1, matrix1, 5), std::logic_error);
}