//
//  multi_online_pitc_gp.cpp
//  GaussianProcess
//
//  Created by Keng Kiat Lim on 9/3/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#include <armadillo>
#include "gtest/gtest.h"
#include "multi_online_pitc_gp.h"

TEST(BasicOperations, InverseReciprocalTest) {
    arma::Mat<double> mat = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    mat.reshape(3, 3);
    arma::Mat<double> inv = 1.0 / mat;
    
    ASSERT_EQ(mat.n_rows, inv.n_rows);
    ASSERT_EQ(mat.n_cols, inv.n_cols);
    
    for (unsigned int row = 0; row < mat.n_rows; ++row) {
        for (unsigned int col = 0; col < mat.n_cols; ++col) {
            ASSERT_FLOAT_EQ(1.0 / mat(row, col), inv(row, col));
        }
    }
}

TEST(BasicOperations, SchurProductTest) {
    arma::Col<double> A = {1, 1, 1, 2};
    arma::Col<double> B = {2, 3, 4, 5};
    arma::Col<double> expected = {2, 3, 4, 10};
    arma::Col<double> actual = A % B;
    
    ASSERT_EQ(A.n_rows, B.n_rows);
    ASSERT_EQ(B.n_rows, expected.n_rows);
    ASSERT_EQ(expected.n_rows, actual.n_rows);
    
    for (unsigned int row = 0; row < expected.n_rows; ++row) {
        ASSERT_FLOAT_EQ(expected(row), actual(row));
    }
}

TEST(BasicOperations, SumTest) {
    arma::Col<double> A = {1, 2, 3, 4};
    auto actual = arma::sum(A);
    auto expected = 10.0;
    
    ASSERT_FLOAT_EQ(expected, actual);
}