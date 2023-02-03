// ==================================================================
// Copyright (c) 2019 Alexander Freed. ALL RIGHTS RESERVED.
// Language: ISO C++14
//
// NeuralNetDigitClassifier class definition
// ==================================================================

#include "NeuralNet.h"

#include "Utility.h"

#include <random>


namespace fnn {


/** Argument Constructor
Sets the number of neurons in the hidden layer.
Initializes the weights.
@param[in] numHidden The number of nodes to put in the hidden layer.
*/
NeuralNetDigitClassifier::NeuralNetDigitClassifier(const unsigned numHidden)
    : m_numHidden(numHidden)
{ }


/** Create weights as a collection of matrices.
Initializes the weights and bias randomly.
Didn't want to use Eigen's setRandom() function because it uses old C++ rand.
@return A set of new matrices of randomly generated weights.
*/
NeuralNetDigitClassifier::WeightsCollection NeuralNetDigitClassifier::generateWeightsRandom() const
{
    std::uniform_real_distribution<> distribution(-0.05, 0.05);
    WeightsCollection weights;
    // weights for input->hidden.
    weights[0] = WeightsType::NullaryExpr(NUM_INPUTS,      m_numHidden, [&distribution]() { return distribution(Global::rng()); });
    // weights for hidden->output
    weights[1] = WeightsType::NullaryExpr(m_numHidden + 1, NUM_OUTPUTS, [&distribution]() { return distribution(Global::rng()); });
    return weights;
}


/** Create weights as a collection of matrices.
Initializes the weights and bias to 0.
@return A set of new matrices of randomly generated weights.
*/
NeuralNetDigitClassifier::WeightsCollection NeuralNetDigitClassifier::generateWeightsZero() const
{
    WeightsCollection weights;
    // dWeights for input->hidden.
    weights[0] = WeightsType::Zero(NUM_INPUTS,      m_numHidden);
    // dWeights for hidden->output
    weights[1] = WeightsType::Zero(m_numHidden + 1, NUM_OUTPUTS);
    return weights;
}


// ------------------------------------------------------------------

/** Feed the input forward and return the selected digit class.
The digit selected is the output node with the highest activation value.
@param[in] inputs A vector of input values.
@param return the chosen digit 0-9.
*/
int NeuralNetDigitClassifier::DetermineDigit(const InputType& inputs) const
{
    // create a place to hold the activation of input->hidden layer
    Eigen::RowVectorXd hiddenActivation(m_numHidden + 1);
    // The bias is the first element. 
    hiddenActivation(0) = 1;
    //Need to map the activation result onto the rest of the holding space
    {
        Eigen::Map<Eigen::RowVectorXd> activationMap(&hiddenActivation(1), m_numHidden);
        activationMap = (inputs * m_weights[0]).unaryExpr(&sigmoid);
    }

    // activate hidden->output layer
    int row, col;
    (hiddenActivation * m_weights[1]).unaryExpr(&sigmoid).maxCoeff(&row, &col);
    return col;
}


// ------------------------------------------------------------------

/** Run the inputs over the weights and adjust the weights if necessary.
@param[in] inputs       One vector of inputs (785)
@param[in] targets      A vector of expected activations (10)
@param[in] learningRate The learning rate.
@param[in] momentum     0 to 1. 0 is equivalent to no momentum. weights += new dWeight + momentum * previous dWeight.
*/
void NeuralNetDigitClassifier::TrainFromInput(const InputType& inputs, const OutputType& targets, const double learningRate, const double momentum)
{
    // create a place to hold the activation of input->hidden layer
    Eigen::RowVectorXd hiddenActivation(m_numHidden + 1);
    // The bias is the first element. 
    hiddenActivation(0) = 1;
    //Need to map the activation result onto the rest of the holding space
    {
        Eigen::Map<Eigen::RowVectorXd> activationMap(&hiddenActivation(1), m_numHidden);
        activationMap = (inputs * m_weights[0]).unaryExpr(&sigmoid);
    }

    // activate hidden->output layer
    const OutputType outputActivation = (hiddenActivation * m_weights[1]).unaryExpr(&sigmoid);

    // calculate error hidden->output
    const auto sigmoidDerivative = [](const double o) { return o * (1 - o); };  // note that input o should already be the output of the sigmoid function. o=Sigmoid(i).
    const OutputType errorOutput = (targets - outputActivation).cwiseProduct(outputActivation.unaryExpr(sigmoidDerivative));

    // calculate error input->hidden
    const Eigen::RowVectorXd errorHidden = (m_weights[1] * errorOutput.transpose()).transpose().cwiseProduct(hiddenActivation.unaryExpr(sigmoidDerivative));

    m_dWeightsPrev[1] = learningRate * hiddenActivation.transpose() * errorOutput + momentum * m_dWeightsPrev[1];
    m_dWeightsPrev[0] = learningRate * inputs.transpose() * Eigen::Map<const Eigen::RowVectorXd>(&errorHidden(1), m_numHidden) + momentum * m_dWeightsPrev[0];
    
    // adjust hidden->output weights
    m_weights[1] += m_dWeightsPrev[1];
    // adjust input->hidden weights
    m_weights[0] += m_dWeightsPrev[0];
}


}
