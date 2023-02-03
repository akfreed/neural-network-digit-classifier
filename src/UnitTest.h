// ==================================================================
// Copyright (c) 2019 Alexander Freed. ALL RIGHTS RESERVED.
// Language: ISO C++14
//
// NeuralNet debugging functions
// ==================================================================

#pragma once

#include <vector>


namespace fnn {
    struct RawTrainer;
}


namespace UnitTest {


bool ValidateLoad(const std::vector<fnn::RawTrainer>& trainingSets, const std::vector<fnn::RawTrainer>& testSets);


}
