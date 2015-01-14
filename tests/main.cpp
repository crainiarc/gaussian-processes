//
//  main.cpp
//  tests
//
//  Created by Keng Kiat Lim on 1/13/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#include <iostream>
#include <armadillo>
#include "gtest/gtest.h"
#import "gp.h"

int main(int argc, char * argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}