# ===================================================================
# Copyright (c) 2019 Alexander Freed
# Language: Python 3.4.4
#
# Saves just a portion of the dataset
# ===================================================================

import csv


def loadCsv(filename):
    data = []
    print("Loading file: {0}".format(filename))
    with open(filename, "r", newline="") as dataFile:
        data = [e for e in csv.reader(dataFile)]
    print("Loaded.")
    return data


def analyze(data):
    targetCount = { str(i):0 for i in range(10)}
    for d in data:
        targetCount[d[0]] += 1
    print("target | count | ratio")
    for pair in [(key, targetCount[key]) for key in sorted(targetCount)]:
        print("    {0}  |  {1} | {2}".format(pair[0], pair[1], round(pair[1] / len(data), 3)))
    print("total: {0}".format(len(data)))


def writeCsv(filename, data):
    print("Saving file: {0}".format(filename))
    with open(filename, "w", newline="") as newDataFile:
        fout = csv.writer(newDataFile)
        for d in data:
            fout.writerow(d)


def main():
    filenameIn = r"../data/mnist_train.csv"
    data = loadCsv(filenameIn)
    analyze(data)

    # get the first half of the data
    data50 = data[:len(data)//2]
    analyze(data50)

    # get a quarter of the data
    # the 2nd quartile is more uniformly distributed
    quarter = len(data) // 4
    data25 = data[quarter*1:quarter*2]
    analyze(data25)

    # write the partial datasets to file
    writeCsv(r"../data/mnist_train50.csv", data50)
    writeCsv(r"../data/mnist_train25.csv", data25)


if __name__ == "__main__":
    main()
