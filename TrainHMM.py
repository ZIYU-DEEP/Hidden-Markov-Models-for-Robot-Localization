from __future__ import print_function
from HMM import *
from DataSet import *
import sys
import pickle



if __name__ == '__main__':

    # This function should be called with one argument: trainingdata.txt
    if (len(sys.argv) < 2 or len(sys.argv) > 2):
        print("Usage: TrainMM.py trainingdata.txt")
        sys.exit(0)

    dataset = DataSet(sys.argv[1])
    dataset.readFile()
    hmm = HMM(dataset.numStates,dataset.numOutputs,dataset.states)
    hmm.train(dataset.obs)

    # Save the model for future use
    print("Saving trained model as ", "trained-model.pkl")
    pickle.dump (hmm, open("trained-model.pkl", "wb"))
