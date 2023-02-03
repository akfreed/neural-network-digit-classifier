// ==================================================================
// Copyright (c) 2019 Alexander Freed. ALL RIGHTS RESERVED.
// Language: ISO C++14
//
// Function prototypes and class definitions for file I/O
// ==================================================================

#pragma once

#include "Trainer.h"

#include <vector>
#include <tuple>


namespace FileIO {


// classes

enum LoadResult
{
    SUCCESS,
    FILE_NOT_FOUND,
    FILE_BAD_FORMAT,
    UNEXPECTED_ERROR
};


// function prototypes

bool CheckLoad(const LoadResult& result);
std::tuple<LoadResult, std::vector<fnn::RawTrainer>> LoadCsv(const std::string& filename, const size_t rowsHint, bool showProgress=false);
std::tuple<LoadResult, std::vector<fnn::RawTrainer>> Deserialize(const std::string& filename);
bool Serialize(const std::string& filename, const std::vector<fnn::RawTrainer>& objects);
void savePlotData(const std::vector<double>& plotData);


}
