# Neural Network

This neural network classifies handwritten digits 0 to 9 in the MNIST dataset. It has one hidden layer. It trains using traditional back-propagation with a momentum factor.

# Setup, Build, and Run

## Linux
1. `mkdir release`
2. `cd release`
3. `cmake .. -DCMAKE_BUILD_TYPE=Release`
4. `make`
5. `./NeuralNet "../data"`

#### Linux without CMake
1. `g++ src/*.cpp -std=c++14 -I eigen/ -o NeuralNet -O2 -march=native`
2. `./NeuralNet "data"`

## Windows with Visual Studio 2017
1. Make folder `build_x64`
2. Open a command prompt in `build_x64`
3. `cmake .. -G "Visual Studio 15 2017 Win64"`
4. Open the solution and set the build to _Release_
5. Build and run

The program is written in ISO C++14

## Usage

`./NeuralNet [dataPath] [numEpochs] [numHidden] [learningRate] [momentum] [defaultSeed] [writePlotData]`

* `dataPath` – Path to data file directory. Type: string. Default: "`../../data/`"
* `numEpochs` – Number of epochs. Type: unsigned. Range: >0. Default: 50
* `numHidden` – Number of nodes in the hidden layer. Type: unsigned. Range: >0. Default: 20
* `learningRate` – The learning rate. Type: double. Range: >0. Default: 0.1
* `momentum` – Coefficient of previous weight change. Range: [0, ~0.97]. Default: 0.9
* `defaultSeed` – Helps with reproducibility when debugging. 1: use default seed. 0: use clock. Default: 0
* `writePlotData` – Write plot data to file "plotdata.csv". 0: don't write. 1: write. Default: 0

# Eigen
This program uses **Eigen**, a C++ header-only library, to do optimized vector and matrix operations. Eigen is open source and licensed mostly under MPL2. Eigen uses column-major order when storing vectors and matrixes. 

# Folder Layout
* data/
    * The _mnist_test.csv_ and _mnist_train.csv_ training data files should go here.
* eigen/
    * The Eigen source code.
* python/
    * _plot.py_ for plotting accuracy and _splitdata.py_ for shortening the datasets.
* src/
    * My Neural Net program source code.

# Class Descriptions
* `NUM_INPUTS` = 785 (defined in _Trainer.h_)
* `NUM_OUTPUTS` = 10 (static member of class `NerualNetDigitClassifier`)
* `InputType` is a typedef for a dynamically sized row-wise vector (defined in _Trainer.h_)
* `OutputType` is a typedef for a matrix with 1 row and `NUM_OUTPUTS` columns
    * Technically a row-wise vector, but was made a matrix to make certain function calls easier. (located in class `NerualNetDigitClassifier`) 
* `WeightsType` is a typedef for a dynamic matrix (located in class `NerualNetDigitClassifier`)
* `WeightsCollection` is a typedef for an array of `WeightsType`, size 2 (located in class `NerualNetDigitClassifier`)

Class `RawTrainer` is a _plain old data_ (“POD”) struct that holds 785 inputs (as an array) and a correct answer (“target”). The first input is the bias input and is always set to 1. `RawTrainer` is used for fast serializing/deserializing. Class `Trainer` also holds 785 inputs and a target, but the inputs are in the form of `InputType` which is usable by the program.

Class `NeuralNetDigitClassifer` has a few members:

* `m_numHidden` is the number of nodes in the hidden layer. This can only be set at construction. 
* `m_weights` is of type `WeightsCollection`, that is a size-2 array of dynamically sized matrixes. The first element is a matrix with 785 rows and `m_numHidden` columns. The second element has `m_numHidden` rows and 10 columns. These are the weights from input->hidden and hidden->output. Every element is initialized randomly
* `m_dWeightsPrev` is the same type as `m_weights`—a size-2 array of matrixes with the same shape as `m_weights`. These hold the previous weight delta for use in calculating the momentum. Every element is initialized to 0.

The class also has some member functions for training. The main ones are `TrainFromInput` and `DetermineDigit`.

Training is sequenced by a function called `train` located in _main.cpp_.

# Neural Network Design

There are 784 inputs +1 for bias. There is one hidden layer with *N* neurons (*N* can be set at run-time). The output layer has 10 neurons. The output with the highest activation is selected as the predicted answer.  

The weights are represented as matrixes. The weights for the input-to-hidden layers are a 785x*N* matrix. The weights for the hidden-to-output layers are a (*N*+1)x10 matrix. The +1 row is for the bias of the hidden-to-output activation, and is always set to 1. The weights are initialized randomly (uniform) in the range _[-0.05, 0.05]_ inclusive. Training is done using back-propagation in stochastic gradient descent with a momentum factor. The training set is shuffled randomly at the beginning of every epoch.

# Program Description
60,000 training inputs are used to train the neural net over 50 epochs. The training inputs are shuffled at the beginning of every epoch. At the end of every epoch the neural net is evaluated for correctness on all 60,000 training inputs as well as 10,000 _test_ inputs that are not used to train. The neural net is also evaluated before any training. After the last epoch, a confusion matrix is created from the test set.

The majority of the work is sequenced in the function named `train` in _main.cpp_ and the `NeuralNetDigitClassifier` member functions in _NeuralNet.cpp_.

## Expected Results

| # Hidden Neurons | Final Accuracy (Test Set) | Approx. Time to Train 50 Epochs |
| ---------------- | ------------------------- | ------------------------------- |
| 10               | ~91%                      | ~1m20s                          |
| 20               | ~94%                      | ~2m47s                          |
| 100              | ~96%                      | ~13m27s                         |

#  License

I lazily slapped on an ALL RIGHTS RESERVED to avoid having to figure out if I'm using GPL code. However, the machine learning algorithm is nothing new and there's nothing particularly special about my code. Feel free to download it and test things out.



-----

Copyright © 2019 Alexander Freed. ALL RIGHTS RESERVED.

Language: Markdown. CommonMark 0.28 compatible.
