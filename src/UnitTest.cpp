// ==================================================================
// Copyright (c) 2019 Alexander Freed. ALL RIGHTS RESERVED.
// Language: ISO C++14
//
// NeuralNet debugging functions
// ==================================================================

#include "UnitTest.h"

#include "NeuralNet.h"
#include "Trainer.h"

#include <cassert>


// macros
#define TEST(condition) if (!(condition)) { assert(false); return false; }


using namespace fnn;


namespace UnitTest {

/** Check a few parts of the data to ensure it was loaded correctly.
@param[in] trainingSets The vector of training data.
@param[in] testSets     The vector of test data.
@return true if the test passed
*/
bool ValidateLoad(const std::vector<fnn::RawTrainer>& trainingSets, const std::vector<fnn::RawTrainer>& testSets)
{
    // training sets
    TEST(trainingSets.size() == 60000);
    {
        auto& first = trainingSets[0];
        // The first input is the bias and should always be 1
        TEST(first.m_inputs[0] == 1);
        // Besides the biast, the first inputs should be 0's at the beginning
        for (int i = 1; i < 153; ++i)
            TEST(first.m_inputs[i] == 0);
        // First non-zero item
        TEST(first.m_inputs[153] == 0.011764705882352941);
        // target should be 5
        TEST(first.m_target == 5);
    }
    {
        auto& second = trainingSets[1];
        TEST(second.m_inputs[0] == 1);
        for (int i = 1; i < 128; ++i)
            TEST(second.m_inputs[i] == 0);
        TEST(second.m_inputs[128] == 0.20000000000000001);
        TEST(second.m_target == 0);
    }
    {
        auto& middle = trainingSets[200];
        TEST(middle.m_inputs[0] == 1);
        for (int i = 1; i < 124; ++i)
            TEST(middle.m_inputs[i] == 0);
        TEST(middle.m_inputs[124] == 0.11372549019607843);
        TEST(middle.m_target == 1);
    }
    {
        auto& middle = trainingSets[49999];
        TEST(middle.m_inputs[0] == 1);
        for (int i = 1; i < 152; ++i)
            TEST(middle.m_inputs[i] == 0);
        TEST(middle.m_inputs[152] == 0.40392156862745099);
        TEST(middle.m_target == 8);
    }
    {
        auto& last = trainingSets[59999];
        TEST(last.m_inputs[0] == 1);
        for (int i = 1; i < 185; ++i)
            TEST(last.m_inputs[i] == 0);
        TEST(last.m_inputs[185] == 0.14901960784313725);
        TEST(last.m_target == 8);
    }

    // test sets
    TEST(testSets.size() == 10000);
    {
        auto& first = testSets[0];
        // The first input is the bias and should always be 1
        TEST(first.m_inputs[0] == 1);
        // First test set should have 0's at the beginning
        for (int i = 1; i < 203; ++i)
            TEST(first.m_inputs[i] == 0);
        // First non-zero item
        TEST(first.m_inputs[203] == 0.32941176470588235);
        // target should be 7
        TEST(first.m_target == 7);
    }
    {
        auto& second = testSets[1];
        TEST(second.m_inputs[0] == 1);
        for (int i = 1; i < 95; ++i)
            TEST(second.m_inputs[i] == 0);
        TEST(second.m_inputs[95] == 0.45490196078431372);
        TEST(second.m_target == 2);
    }
    {
        auto& middle = testSets[250];
        TEST(middle.m_inputs[0] == 1);
        for (int i = 1; i < 151; ++i)
            TEST(middle.m_inputs[i] == 0);
        TEST(middle.m_inputs[151] == 0.031372549019607843);
        TEST(middle.m_target == 4);
    }
    {
        auto& last = testSets[9999];
        TEST(last.m_inputs[0] == 1);
        for (int i = 1; i < 74; ++i)
            TEST(last.m_inputs[i] == 0);
        TEST(last.m_inputs[74] == 0.031372549019607843);
        TEST(last.m_target == 6);
    }

    return true;
}


}
