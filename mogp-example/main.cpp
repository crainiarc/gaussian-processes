//
//  main.cpp
//  MOGPExample
//
//  Created by Keng Kiat Lim on 10/3/15.
//  Copyright (c) 2015 Keng Kiat Lim. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <armadillo>
#include "../gaussian_processes/multi_output.h"

#define M_BLOCK_SIZE 90

auto readHyperFromFile(const char *fileName) -> MultiOutputHyperparameters;
auto readTrainingFile(const char *fileName) -> std::tuple<arma::Cube<double>, arma::Cube<double>, arma::Col<double>>;
auto readTestFile(const char *fileName) -> std::tuple<arma::Cube<double>, arma::Cube<double>>;

auto main(int argc, const char * argv[]) -> int {
    if (argc != 5) {
        std::cout << "Invalid number of arguments.\n";
        exit(0);
    }
    
    auto kernFile = argv[1];
    auto trainingFile = argv[2];
    auto testFile = argv[3];
    auto outputFile = argv[4];
    auto blockSize = std::stoi(argv[5]);
    
    try {
        auto hypers = readHyperFromFile(kernFile);
        auto trainTuple = readTrainingFile(trainingFile);
        auto trainingSet = std::get<0>(trainTuple);
        auto observations = std::get<1>(trainTuple);
        auto latentVariables = std::get<1>(trainTuple);
        
        auto testTuple = readTestFile(testFile);
        auto XTest = std::get<0>(testTuple);
        auto YTest = std::get<0>(testTuple);
        
        auto file = std::fstream(outputFile, std::ios::out);
        if (!file.is_open()) {
            throw std::string("Unable to open file");
        }
        
        auto mogp = MultiOutputOnlinePITCGP(blockSize, latentVariables, hypers);
        for (auto i = 0; i < trainingSet.n_slices; ++i) {
            mogp.addTrainingSet(trainingSet.slice(i), observations.slice(i));
            auto predictTuple = mogp.predict(XTest.slice(i));
            
            for (auto j = 0; j < std::get<0>(predictTuple).n_rows; ++j) {
                file << std::get<0>(predictTuple)(j) << ", " << std::get<1>(predictTuple)(j) << "\n";
            }
        }
        
        file.close();
    } catch (std::string errMsg) {
        std::cout << errMsg << std::endl;
    }
    
    return 0;
}

auto readHyperFromFile(const char *fileName) -> MultiOutputHyperparameters {
    auto hypers = MultiOutputHyperparameters();
    auto file = std::fstream(fileName, std::ios::in);
    
    if (!file.is_open()) {
        throw std::string("Unable to open file");
    }
    
    int inputDimension, outputDimension;
    file >> inputDimension >> outputDimension;
    
    // Resize all matrices in hypers
    hypers.biases.resize(outputDimension + 1);
    hypers.scales.resize(outputDimension + 1);
    hypers.betas.resize(outputDimension + 1);
    hypers.variances.resize(outputDimension + 1);
    hypers.sigma2Ys.resize(outputDimension + 1);
    hypers.sigma2Us.resize(outputDimension + 1);
    hypers.precisionYs.resize(outputDimension + 1, inputDimension);
    hypers.precisionUs.resize(outputDimension + 1, inputDimension);
    
    file.ignore(256, ' ');
    file >> hypers.sigma2Us(0);
    for (auto i = 0; i < inputDimension; ++i) {
        file >> hypers.precisionUs(0, i);
    }
    
    for (auto i = 1; i < outputDimension + 1; ++i){
        file.ignore(256, ' ');
        file >> hypers.sigma2Us(i) >> hypers.sigma2Ys(i);
        
        for (auto j = 0; j < inputDimension; ++j) {
            file >> hypers.precisionUs(i, j);
        }
        
        for (auto j = 0; j < inputDimension; ++j) {
            file >> hypers.precisionYs(i, j);
        }
    }
    
    for (auto i = 0; i < outputDimension + 1; ++i) {
        file.ignore(256, ' ');
        file >> hypers.variances(i);
    }
    
    for (auto i = 1; i < outputDimension + 1; ++i) {
        file >> hypers.biases(i);
    }
    
    for (auto i = 1; i < outputDimension + 1; ++i) {
        file >> hypers.scales(i);
    }
    
    for (auto i = 1; i < outputDimension + 1; ++i) {
        file >> hypers.betas(i);
    }
    
    file.close();
    return hypers;
}

auto readTrainingFile(const char *fileName) -> std::tuple<arma::Cube<double>, arma::Cube<double>, arma::Col<double>> {
    auto X = arma::Cube<double>();
    auto Y = arma::Cube<double>();
    auto latentVars = arma::Col<double>();
    
    auto file = std::fstream(fileName, std::ios::in);
    if (!file.is_open()) {
        throw std::string("Unable to open file");
    }
    
    int numLatentVars;
    file >> numLatentVars;
    latentVars.resize(numLatentVars);
    
    int numTimeSteps, blockSize, inputDimensions, outputDimensions;
    file >> numTimeSteps >> blockSize >> inputDimensions >> outputDimensions;
    X.resize(inputDimensions, blockSize * outputDimensions, numTimeSteps);
    Y.resize(outputDimensions, blockSize, numTimeSteps);
    
    for (auto i = 0; i < X.n_slices; ++i) {
        for (auto j = 0; j < X.n_rows; ++j) {
            for (auto k = 0; k < X.n_cols; ++k) {
                file >> X(j, k, i);
            }
        }
    }
    
    for (auto i = 0; i < Y.n_slices; ++i) {
        for (auto j = 0; j < Y.n_rows; ++j) {
            for (auto k = 0; k < Y.n_cols; ++k) {
                file >> X(j, k, i);
            }
        }
    }
    
    file.close();
    return std::make_tuple(X, Y, latentVars);
}

auto readTestFile(const char *fileName) -> std::tuple<arma::Cube<double>, arma::Cube<double>> {
    auto X = arma::Cube<double>();
    auto Y = arma::Cube<double>();
    
    auto file = std::fstream(fileName, std::ios::in);
    if (!file.is_open()) {
        throw std::string("Unable to open file");
    }
    
    int numTimeSteps, testCaseSize, inputDimensions, outputDimensions;
    file >> numTimeSteps >> testCaseSize >> inputDimensions >> outputDimensions;
    X.resize(inputDimensions, testCaseSize * outputDimensions, numTimeSteps);
    Y.resize(outputDimensions, testCaseSize, numTimeSteps);
    
    for (auto i = 0; i < X.n_slices; ++i) {
        for (auto j = 0; j < X.n_rows; ++j) {
            for (auto k = 0; k < X.n_cols; ++k) {
                file >> X(j, k, i);
            }
        }
    }
    
    for (auto i = 0; i < Y.n_slices; ++i) {
        for (auto j = 0; j < Y.n_rows; ++j) {
            for (auto k = 0; k < Y.n_cols; ++k) {
                file >> X(j, k, i);
            }
        }
    }
    
    file.close();
    return std::make_tuple(X, Y);
}