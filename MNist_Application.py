import numpy as np
import random as rm
import matplotlib.pyplot as plt
import NeuralNetwork as nn

class MNist():
    
    def __init__(self):
        self.ims_train = np.load('images_train.npy')
        self.lbs_train = np.load('labels_train.npy')
        self.ims_test = np.load('images_test.npy')
        self.lbs_test = np.load('labels_test.npy')

def main():

    NN = nn.Network(layers=(10, 5, 5, 3), activation=nn.Activation.sigmoid)
    NN.predict(np.random.rand(10, 1))

if __name__ == "__main__":
    main()