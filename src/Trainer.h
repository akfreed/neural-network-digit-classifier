// ==================================================================
// Copyright (c) 2019 Alexander Freed. ALL RIGHTS RESERVED.
// Language: ISO C++14
//
// The class definition for Trainer and RawTrainer
// ==================================================================

#pragma once

#include <array>
#include <Eigen/Dense>


namespace fnn {


// ------------------------------------------------------------------

constexpr unsigned NUM_INPUTS = 785;  // 28*28 = 184. +1 for bias
using InputType = Eigen::RowVectorXd;


// ------------------------------------------------------------------

/** Used for serializing and deserializing the training or test sets
*/
struct RawTrainer
{
    int                             m_target;
    std::array<double, NUM_INPUTS> m_inputs;
};


// ------------------------------------------------------------------

/** Holds a training/test input and expected value in a form usable by Eigen
*/
class Trainer
{
public:
    Trainer() = default;

    /** Argument constructor
    @param[in] target    The correct answer for the training inputs.
    @param[in] pInputs   An array of inputs.
    @param[in] numInputs The length of the input array.
    */
    Trainer(const int target, const double* const pInputs, const size_t numInputs)
        : m_target(target)
        , m_inputs(Eigen::Map<const InputType>(pInputs, numInputs))
    { }

    /** Construct from a RawTrainer object.
    Copies the RawTrainer.
    @param[in] rawTrainer the RawTrainer to be copied.
    */
    explicit Trainer(const RawTrainer& rawTrainer)
        : Trainer(rawTrainer.m_target, rawTrainer.m_inputs.data(), rawTrainer.m_inputs.size())
    { }

    int GetTarget() const { return m_target; } 
    const InputType& GetInputs() const { return m_inputs; }

private:
    int       m_target;
    InputType m_inputs;
};


}
