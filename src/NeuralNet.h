// ==================================================================
// Copyright (c) 2019 Alexander Freed. ALL RIGHTS RESERVED.
// Language: ISO C++14
//
// NeuralNetDigitClassifier class declaration
// ==================================================================

#pragma once

#include "Trainer.h"

#include <array>

#include <Eigen/Dense>


namespace fnn {


/** A neural network with 1 hidden layer.
*/
class NeuralNetDigitClassifier
{
public:
    // static consts
    constexpr static unsigned NUM_OUTPUTS = 10;
    // public typedefs
    using WeightsType       = Eigen::MatrixXd;
    using OutputType        = Eigen::Matrix<double, 1, NUM_OUTPUTS>;
    using WeightsCollection = std::array<WeightsType, 2>;

    // public functions
    NeuralNetDigitClassifier() = default;
    explicit NeuralNetDigitClassifier(const unsigned numHidden);

    int  DetermineDigit(const InputType& inputs) const;
    void TrainFromInput(const InputType& inputs, const OutputType& targets, const double learningRate, const double momentum);

private:
    // private functions
    static double sigmoid(const double z) { return 1.0 / (1.0 + exp(-z)); };
    WeightsCollection generateWeightsRandom() const;
    WeightsCollection generateWeightsZero() const;

    // private data
    unsigned          m_numHidden    = 20;
    WeightsCollection m_weights      = generateWeightsRandom();
    WeightsCollection m_dWeightsPrev = generateWeightsZero();
};


}
