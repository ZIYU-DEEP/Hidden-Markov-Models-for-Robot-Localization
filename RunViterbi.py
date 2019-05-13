from __future__ import print_function
import sys
from DataSet import *
from HMM import *
import numpy as np

import pickle

if __name__ == '__main__':

    # This function can be called with just the test data or the test data along with a saved model
    if (len(sys.argv) < 2 or len(sys.argv) > 3):
        print("Usage: TrainMM.py testingdata.txt trained-model.pkl")
        sys.exit(0)

    print ("Loading observation sequence from file ", sys.argv[1])
    dataset = DataSet(sys.argv[1])
    dataset.readFile()

    if (len(sys.argv) == 2):
        print("No model provided. Using default HMM model")
        hmm = HMM(dataset.numStates,dataset.numOutputs,dataset.states)
    else:
        print("Loading HMM model from ", sys.argv[2])
        hmm = pickle.load (open (sys.argv[2], "rb"))

    AllXS = hmm.viterbi(dataset.obs)


    # Compare this with the ground-truth
    numCorrect = 0
    numIncorrect = 0;
    for i in range(len(dataset.obs)):
        correct = np.sum(np.equal(AllXS[i], dataset.states[i]))
        incorrect = len(dataset.states[i]) - correct
        numCorrect = numCorrect + correct
        numIncorrect = numIncorrect + incorrect

    print('Total number of correct: ', numCorrect, '({0:.2f}'.format(100*numCorrect/(numCorrect+numIncorrect)), 'percent), Total number of incorrect: ', numIncorrect)
