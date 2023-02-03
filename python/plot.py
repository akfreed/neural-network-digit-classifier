# ===================================================================
# Copyright (c) 2019 Alexander Freed
# Language: Python 3.4.4
#
# Plots accuracy
#
# https://pythonprogramming.net/loading-file-data-matplotlib-tutorial/
# ===================================================================

import matplotlib.pyplot as plt
import csv
import glob


def plot(filename):
    epoch       = []
    accTraining = []
    accTest     = []

    with open(filename, 'r') as file:
        plots = csv.reader(file, delimiter=',')
        for row in plots:
            epoch.append(int(row[0]))
            accTraining.append(float(row[1]))
            accTest.append(float(row[2]))

    plt.clf()
    plt.plot(epoch, accTraining, label='Training Inputs')
    plt.plot(epoch, accTest, label='Test Inputs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy/Epoch')
    plt.legend()


def plotWithCloseup(filename):
    # save the regular plot
    plot(filename)
    plt.savefig(filename + ".png")
    # save the close-up plot
    plt.ylim(.88, 1)
    plt.savefig(filename + "_close.png")


def plotAll():
    files = glob.glob('*.csv_*')
    for file in files:
        plotWithCloseup(file)


if __name__ == "__main__":
    plotAll()
