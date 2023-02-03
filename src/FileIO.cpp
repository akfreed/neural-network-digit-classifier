// ==================================================================
// Copyright (c) 2019 Alexander Freed. ALL RIGHTS RESERVED.
// Language: ISO C++14
//
// Contains the loading code for file I/O.
// ==================================================================

#include "FileIO.h"

#include <fstream>
#include <cassert>
#include <iostream>


namespace FileIO {


/** Check the load result and print out a helpful message.
@param[in] result The result to check.
@return true if the load result was a success
*/
bool CheckLoad(const LoadResult& result)
{
    switch (result)
    {
    case LoadResult::SUCCESS:
        return true;
        break;
    case LoadResult::FILE_NOT_FOUND:
        std::cout << "File does not exist.\n";
        break;
    case LoadResult::FILE_BAD_FORMAT:
        std::cout << "File format did not match expectations.\n";
        break;
    case LoadResult::UNEXPECTED_ERROR:
        std::cout << "Unexpected error.\n";
        break;
    default:
        // option not accounted for
        assert(false);
        break;
    }
    return false;
}


/** Load the data from a CSV.
This is slower than derserializing, but portable.
@param[in] filename     The path and filename
@param[in] rowsHint     The expected number of rows. Used to preallocate space.
@param[in] showProgress [default: false] If true, print to stdout to show progress.
@return A pair consisting of a load result and a std::vector of RawTrainer objects
*/
std::tuple<LoadResult, std::vector<fnn::RawTrainer>> LoadCsv(const std::string& filename, const size_t rowsHint, bool showProgress)
{
    // open file
    std::fstream fin(filename.c_str(), std::ios::in);
    if (!fin)
        return { LoadResult::FILE_NOT_FOUND, {} };

    // allocate object array
    std::vector<fnn::RawTrainer> objects;
    objects.reserve(rowsHint);

    // begin loading

    // create a default-constructed object
    objects.emplace_back();
    // read in the target
    fin >> objects[0].m_target;
    // read in the comma
    char delimiter;
    fin >> delimiter;

    // load from file
    while (fin && !fin.eof())
    {
        if (showProgress && objects.size() % 5000 == 0)
            std::cout << "Loaded: " << objects.size() << std::endl;

        auto& last = objects.back();

        // read in the inputs
        for (auto& elem : last.m_inputs)
        {
            // skip first element, which is the bias (not saved in the CSV)
            if (&elem == &last.m_inputs.front())
                continue;

            if (!fin || fin.eof() || delimiter != ',')
                break;

            fin >> elem;
            // whitespace for the newline after last element is automatically removed
            if (&elem != &last.m_inputs.back())
                fin >> delimiter;
        }

        // eof should not be triggered yet or the file is the wrong size
        if (!fin || fin.eof() || delimiter != ',')
        {
            // should be a target followed by 784 values (comma separated) on each line
            assert(false);
            return { LoadResult::FILE_BAD_FORMAT, {} };
        }

        
        // create a default-constructed object
        objects.emplace_back();

        // read in the target
        fin >> objects.back().m_target;
        // read in the comma
        fin >> delimiter;
    }
    
    objects.pop_back();  // get rid of last dummy object

    if (showProgress && objects.size() % 5000 != 0)
        std::cout << "Loaded: " << objects.size() << std::endl;

    return { LoadResult::SUCCESS, std::move(objects) };
}


/** Deserialize the data from a file.
Assumes a binary file containing serialized TrainingSet objects.
This is faster but very non-portable! A machine should be able to deserialize a file it has serialized itself.
@param[in] filename The path and filename
@return A pair consisting of a load result and a std::vector of RawTrainer objects
*/
std::tuple<LoadResult, std::vector<fnn::RawTrainer>> Deserialize(const std::string& filename)
{
    // open file
    std::fstream fin(filename.c_str(), std::ios::binary | std::ios::in);
    if (!fin)
        return { LoadResult::FILE_NOT_FOUND, {} };

    // calculate necessary number of objects
    const auto beginPos = fin.tellg();
    fin.seekg(0, std::ios::end);
    const std::ios::pos_type fileBytes = fin.tellg() - beginPos;
    if (fileBytes % sizeof(fnn::RawTrainer) != 0)
    {
        assert(false);
        return { LoadResult::FILE_BAD_FORMAT, {} };
    }
    const unsigned numObjects = unsigned(fileBytes / sizeof(fnn::RawTrainer));
    fin.seekg(0);

    // allocate object array
    std::vector<fnn::RawTrainer> objects(numObjects);

    // peek triggers eof
    fin.peek();
    // load from file
    if (fin && !fin.eof())
        fin.read(reinterpret_cast<char*>(&objects[0]), numObjects * sizeof(fnn::RawTrainer));

    assert(!fin.eof());

    // peek triggers eof
    fin.peek();
    
    // sanity checks
    if (fin.fail() || !fin.eof())
    {
        assert(false);
        return { LoadResult::UNEXPECTED_ERROR, {} };
    }

    return { LoadResult::SUCCESS, std::move(objects) };
}


/** Serialize the data.
Creates a binary file containing serialized TrainingSet objects.
The binary file is faster to load than a CSV, but this is very non-portable! However, a machine should be able to deserialize a file it has serialized itself.
@param[in] filename The path and filename
@param[in] objects  A std::vector of objects to serialize
@return true if successful
*/
bool Serialize(const std::string& filename, const std::vector<fnn::RawTrainer>& objects)
{
    // open file
    std::fstream fout(filename.c_str(), std::ios::binary | std::ios::out);
    if (!fout)
        return false;

    // write to file
    fout.write(reinterpret_cast<const char*>(&objects[0]), objects.size() * sizeof(fnn::RawTrainer));

    // sanity checks
    if (fout.fail())
    {
        assert(false);
        return false;
    }

    return true;
}


/** Save the plot data to a file.
For now the path is hard coded and the plotData vector format is a little weird.
@param[in] plotData The vector of accuracy data. Even indices are accuracy values for training data. Odd indices are accuracy values for test data.
*/
void savePlotData(const std::vector<double>& plotData)
{
    if (plotData.size() % 2 != 0)
    {
        assert(false);
        return;
    }

    const std::string path = "plotdata.csv";

    // decode and reorder the plotData
    // even indices are Training points, odd are Test.
    std::fstream fout(path.c_str(), std::ios::out);

    for (size_t i = 0; i < plotData.size(); i += 2)
        fout << i / 2 << ',' << plotData[i] << ',' << plotData[i + 1] << std::endl;
}


}
