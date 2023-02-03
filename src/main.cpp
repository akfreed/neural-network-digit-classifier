// ==================================================================
// Copyright (c) 2019 Alexander Freed. ALL RIGHTS RESERVED.
// Language: ISO C++14
//
// Program entry.
// Sequences the neural network training.
// ==================================================================

#include "FileIO.h"
#include "NeuralNet.h"
#include "UnitTest.h"
#include "Utility.h"

#include <iostream>
#include <ios>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cassert>


using namespace fnn;


// ------------------------------------------------------------------
// loading / saving

/** Do some pre-processing on the data.
Normalize to a 0..1 range
set the biases to 1.0
@param[in/out] trainers The training data to process.
@return false if any training data failed the range check.
*/
bool preprocess(std::vector<RawTrainer>& trainers)
{
    // sanity check. Data should be in range [0, 255] inclusive
    for (auto& trainer : trainers)
    {
        decltype(trainer.m_inputs)::const_iterator min, max;
        std::tie(min, max) = std::minmax_element(trainer.m_inputs.begin(), trainer.m_inputs.end());

        if (trainer.m_target < 0 || trainer.m_target > 9 || min == trainer.m_inputs.cend() || *min < 0 || *max > 255)
        {
            assert(false);
            return false;
        }
    }
    
    // 1) normalize data to a 0..1 range
    // 2) set bias (first weight) to 1.0
    std::for_each(trainers.begin(), trainers.end(), 
        [](RawTrainer& trainer) { 
            trainer.m_inputs[0] = 1.0;
            const auto firstWeight = std::next(trainer.m_inputs.begin());  // the first weight is the second element
            std::transform(firstWeight, trainer.m_inputs.end(), firstWeight, [](const double val) { return val / 255.0; });
        });

    return true;
}


/** load the training and test sets
@param[out] out_trainingSet An output vector of loaded training set data
@param[out] out_testSet     An output vector of loaded test set data
@param true if load was successful.
*/
bool load(const std::string& basePath, std::vector<Trainer>& out_trainingSet, std::vector<Trainer>& out_testSet)
{
    // hard-code the filenames
    const std::string pathTrainingSet       = basePath + "mnist_train.csv";
    const std::string pathTestSet           = basePath + "mnist_test.csv";
    const std::string pathTrainingProcessed = basePath + "mnist_train.bin";
    const std::string pathTestProcessed     = basePath + "mnist_test.bin";

    FileIO::LoadResult result = FileIO::LoadResult::UNEXPECTED_ERROR;
    std::vector<RawTrainer> rawTrainingSet;
    std::vector<RawTrainer> rawTestSet;
    bool mustLoadCsv = false;

    // first try to load the preprocessed data. If this is the first time the program is run
    // on this machine, this will fail.
    std::cout << "Loading preprocessed data.\n";
    std::cout << "Loading: " << pathTrainingProcessed << std::endl;
    std::tie(result, rawTrainingSet) = FileIO::Deserialize(pathTrainingProcessed);
    // handle I/O errors
    if (!FileIO::CheckLoad(result))
        mustLoadCsv = true;
    std::cout << "Loading: " << pathTestProcessed << std::endl;
    std::tie(result, rawTestSet) = FileIO::Deserialize(pathTestProcessed);
    if (!FileIO::CheckLoad(result))
        mustLoadCsv = true;

    // if we couldn't load the preprocessed data, load the regular CSV's, process them, then save them to disk.
    if (!mustLoadCsv)
        std::cout << "Preprocessed data successfully loaded." << std::endl;
    else
    {
        std::cout << "Unable to load preprocessed data. Must load data from CSV.\n"
                 <<  "This may take ~30 seconds in a release build and ~4 minutes in a debug build.\n"
                  << "Binary files will be generated in the same directory to speed up future loading.\n";

        std::cout << "Loading: " << pathTrainingSet << std::endl;
        std::tie(result, rawTrainingSet) = FileIO::LoadCsv(pathTrainingSet, 60000, true);
        // handle I/O errors
        if (!FileIO::CheckLoad(result))
        {
            std::cout << "Unable to load file: " << pathTrainingSet << std::endl;
            return false;
        }
        std::cout << "Loading: " << pathTestSet << std::endl;
        std::tie(result, rawTestSet) = FileIO::LoadCsv(pathTestSet, 10000, true);
        if (!FileIO::CheckLoad(result))
        {
            std::cout << "Unable to load file: " << pathTestSet << std::endl;
            return false;
        }

        // run preprocessing
        std::cout << "Processing data...";
        std::cout.flush();
        if (!preprocess(rawTrainingSet) || !preprocess(rawTestSet))
        {
            std::cout << "Failed!\nData was formatted incorrectly." << std::endl;
            return false;
        }
        std::cout << "Done." << std::endl;

        // save the processed training data for faster loading next time
        std::cout << "Saving processed data for faster load next time...";
        std::cout.flush();
        if (FileIO::Serialize(pathTrainingProcessed, rawTrainingSet) &&
            FileIO::Serialize(pathTestProcessed, rawTestSet))
        {
            std::cout << "Done." << std::endl;
        }
        else
            std::cout << "Failed!\nUnable to save processed data. Program can still continue." << std::endl;
    }

    // validate load
    std::cout << "Validating load...";
    std::cout.flush();
    if (!UnitTest::ValidateLoad(rawTrainingSet, rawTestSet))
    {
        std::cout << "Failed!\nLoad unsuccessful." << std::endl;
        return false;
    }
    std::cout << "Done." << std::endl;

    // convert the sets
    std::cout << "Converting data into internal representation...";
    std::cout.flush();
    out_trainingSet.clear();
    out_testSet.clear();

    out_trainingSet.reserve(rawTrainingSet.size());
    out_testSet.reserve(rawTestSet.size());

    for (auto& trainer : rawTrainingSet)
        out_trainingSet.emplace_back(trainer);
    for (auto& trainer : rawTestSet)
        out_testSet.emplace_back(trainer);

    std::cout << "Done" << std::endl;

    return true;
}


// ==================================================================
// training

/** Evaluate the neural network with a whole collection of training data to check accuracy.
Calculate the ratio of correct answers / total inputs
@param[in] neuralnet The neural net object.
@param[in] data      A standard container of trainers.
@return The ratio of correct answers / total inputs.
*/
template <typename DataContainer>
double Evaluate(const NeuralNetDigitClassifier& neuralnet, const DataContainer& data)
{
    int correct = 0;
    for (auto& trainer : data)
    {
        const int answer = neuralnet.DetermineDigit(trainer.GetInputs());
        if (answer == trainer.GetTarget())
            ++correct;
    }
    return correct / static_cast<double>(data.size());
}


/** Calls Evaluate on the training data and test data.
@param[in]     neuralnet   The neural net object.
@param[in]     trainingSet The vector of training data.
@param[in]     testSet     The vector of test data.
@param[in/out] plotData    A vector to hold data for plotting later.
*/
template <typename DataContainer>
void EvaluateWrapper(const NeuralNetDigitClassifier& neuralnet, const DataContainer& trainingSet, const DataContainer& testSet, std::vector<double>& plotData)
{
    const double accuracyTraining = Evaluate(neuralnet, trainingSet);
    std::cout << "    Training Set Accuracy : " << accuracyTraining * 100 << "%" << std::endl;
    const double accuracyTest = Evaluate(neuralnet, testSet);
    std::cout << "    Test Set Accuracy     : " << accuracyTest * 100 << "%" << std::endl;

    plotData.push_back(accuracyTraining);
    plotData.push_back(accuracyTest);
}


/** Evaluate the neural network with a whole collection of training data to check accuracy.
Calculate the ratio of correct answers / total inputs
@param[in] neuralnet The neural net object.
@param[in] data      A standard container of trainers.
@return The ratio of correct answers / total inputs.
*/
template <typename DataContainer>
Eigen::MatrixXd BuildConfusionMatrix(const NeuralNetDigitClassifier& neuralnet, const DataContainer& data)
{
    Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero(NeuralNetDigitClassifier::NUM_OUTPUTS, NeuralNetDigitClassifier::NUM_OUTPUTS);

    for (auto& trainer : data)
    {
        // row index (y): correct answer
        // col index (x): given answer
        const int answer = neuralnet.DetermineDigit(trainer.GetInputs());
        confusionMatrix(trainer.GetTarget(), answer) += 1;
    }
    return confusionMatrix;
}


/** Train the neuralnet.
@param[in] trainingSet    The vector of training data. Pass by move (with std::move) because it gets shuffled.
@param[in] testSet        The vector of test data. Pass by move (with std::move) because it gets shuffled.
@param[in] numEpochs      The number of epochs to run.
@param[in] numHiddenNodes The number of nodes in the hidden layer.
@param[in] learningRate   The learning rate.
@param[in] momentum       The momentum. 0 to 1. 0 is equivalent to no momentum.
@param[in] writePlotData  [default: false] true to save the accuracy data to a file for plotting later.
*/
void train(std::vector<Trainer>&& trainingSet, 
           std::vector<Trainer>&& testSet, 
           const unsigned numEpochs, 
           const unsigned numHiddenNodes,
           const double   learningRate, 
           const double   momentum, 
           const bool     writePlotData=false)
{
    // display training params
    const auto displayParams = [numHiddenNodes, learningRate, momentum]() {
        std::cout << "\n"
                  << "Training Parameters:\n"
                  << "    num hidden nodes = " << numHiddenNodes << "\n"
                  << "    learning rate = " << learningRate << "\n"
                  << "    momentum = " << momentum << "\n"
                  << "    random seed = 0x" << std::hex << Global::get_seed() << std::dec << std::endl;
    };
    displayParams();

    // init neural net
    NeuralNetDigitClassifier neuralnet(numHiddenNodes);

    std::vector<double> plotData;

    // check initial accuracy
    std::cout << "\nInitial accuracy evaluation..." << std::endl;
    EvaluateWrapper(neuralnet, trainingSet, testSet, plotData);

    // for every epoch...
    for (unsigned epochIndex = 0; epochIndex < numEpochs; ++epochIndex)
    {
        // shuffle the training set
        std::shuffle(trainingSet.begin(), trainingSet.end(), Global::rng());

        NeuralNetDigitClassifier::OutputType targets(10);
        
        // for every training input...
        for (auto& trainer : trainingSet)
        {
            // set the expted target for this input
            targets.setConstant(0.1);
            targets(trainer.GetTarget()) = 0.9;
            // call the neural net training routine
            neuralnet.TrainFromInput(trainer.GetInputs(), targets, learningRate, momentum);
        }

        // evaluate
        std::cout << "\nEnd of Epoch " << epochIndex + 1 << " of " << numEpochs << ". Evaluating accuracy..." << std::endl;
        EvaluateWrapper(neuralnet, trainingSet, testSet, plotData);
    }

    // save plot data
    if (writePlotData)
        FileIO::savePlotData(plotData);

    // display training params again
    displayParams();

    // display confusion matrix
    const Eigen::MatrixXd confusionMatrix = BuildConfusionMatrix(neuralnet, testSet);
    std::cout << "\nConfusion Matrix\n"
              << "    y-axis=correct answer\n"
              << "    x-axis=guessed answer\n"
              << confusionMatrix << std::endl;
}


// ==================================================================
// parse args

/** Print the usage.
*/
void displayHelp()
{
    std::cout << "Usage:\n"
              << "./NeuralNet [dataPath] [numEpochs] [numHidden] [learningRate] [momentum] [defaultSeed] [writePlotData]\n\n" 
              << "    dataPath      - Path to data file directory. Type: string. Default: \"../../data/\"\n"
              << "    numEpochs     - Number of epochs. Type: unsigned. Range: >0. Default: 50\n"
              << "    numHidden     - Number of nodes in the hidden layer. Type: unsigned. Range: >0. Default: 20\n"
              << "    learningRate  - The learning rate. Type: double. Range: >0. Default: 0.1\n"
              << "    momentum      - Coefficient of previous weight change. Range: [0, ~0.97]. Default: 0.9\n"
              << "    defaultSeed   - Helps with reproducibility when debugging. 1: use default seed. 0: use clock. Default: 0\n"
              << "    writePlotData - Write plot data to file \"plotdata.csv\". 0: don't write. 1: write. Default: 0\n"
              << std::endl;
}


/** parse the command line arguments
@param[in] argc Length of argv.
@param[in] argv An array of arguments.
@return A tuple of command-line settings. (basePath, numEpochs, numHidden, learningRate, momentum, writePlotData, valid).
*/
std::tuple<std::string, unsigned, unsigned, double, double, bool, bool> parseArgs(int argc, char** argv)
{
    // set defaults
    std::string basePath = R"(../../data/)";
    unsigned numEpochs   = 50;
    unsigned numHidden   = 20;
    double learningRate  = 0.1;
    double momentum      = 0.9;
    bool writePlotData   = false;

    bool valid           = true;

    // path to files
    if (argc > 1)
    {
        basePath = argv[1];
        if (basePath.back() != '/' && basePath.back() != '\\')
            basePath += '/';
    }
    // number of epochs
    if (argc > 2)
    {
        try
        {
            numEpochs = std::stoul(argv[2]);
        }
        catch (...)
        {
            std::cout << "Unable to parse argument 2: " << argv[2] << "\n";
            valid = false;
        }
    }
    // number of nodes in hidden layer
    if (argc > 3)
    {
        try
        {
            numHidden = std::stoul(argv[3]);
        }
        catch (...)
        {
            std::cout << "Unable to parse argument 3: " << argv[3] << "\n";
            valid = false;
        }
    }
    // learning rate
    if (argc > 4)
    {
        try
        {
            learningRate = std::stod(argv[4]);
        }
        catch (...)
        {
            std::cout << "Unable to parse argument 4: " << argv[4] << "\n";
            valid = false;
        }
    }
    // momentum
    if (argc > 5)
    {
        try
        {
            momentum = std::stod(argv[5]);
        }
        catch (...)
        {
            std::cout << "Unable to parse argument 5: " << argv[5] << "\n";
            valid = false;
        }
    }
    // use debugging seed. Helps with repeatability.
    if (argc > 6)
    {
        try
        {
            const int useDebugging = std::stoi(argv[6]);
            if (useDebugging != 0)
                Global::seed_default();
        }
        catch (...)
        {
            std::cout << "Unable to parse argument 6: " << argv[6] << "\n";
            valid = false;
        }
    }
    // write plot data
    if (argc > 7)
    {
        try
        {
            writePlotData = (std::stoi(argv[7]) != 0);
        }
        catch (...)
        {
            std::cout << "Unable to parse argument 7: " << argv[7] << "\n";
            valid = false;
        }
    }

    if (!valid)
    {
        std::cout << "\n";
        displayHelp();
    }

    return std::make_tuple(basePath, numEpochs, numHidden, learningRate, momentum, writePlotData, valid);
}


// ==================================================================
// main

int main(int argc, char** argv)
{
    // parse args
    std::string basePath;
    unsigned numEpochs;
    unsigned numHidden;
    double learningRate;
    double momentum;
    bool writePlotData;
    bool validArgs;
    std::tie(basePath, numEpochs, numHidden, learningRate, momentum, writePlotData, validArgs) = parseArgs(argc, argv);
    if (!validArgs)
        return EXIT_FAILURE;

    // load the data
    std::vector<Trainer> trainingSet;
    std::vector<Trainer> testSet;
    if (!load(basePath, trainingSet, testSet))
    {
        displayHelp();
        return EXIT_FAILURE;
    }
    
    // train
    train(std::move(trainingSet), std::move(testSet), numEpochs, numHidden, learningRate, momentum, writePlotData);

    std::cout << "\nEnd of program." << std::endl;
    return EXIT_SUCCESS;
}


// ==================================================================
