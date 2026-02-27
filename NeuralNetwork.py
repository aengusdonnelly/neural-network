import numpy as np

class Network():

    def __init__(self, layers, activation):

        self.ls = layers
        self.Ws = []
        self.bs = []
        self.activation = activation

        for i in range(1, len(self.ls)):
            W = np.random.rand(self.ls[i], self.ls[i-1])
            b = np.random.rand(self.ls[i], 1)
            self.Ws.append(W)
            self.bs.append(b)

    def train(self):
        #TODO: Write a method for back-propagation.
        pass

    def predict(self, a):
        
        for i in range(len(self.Ws)):
            a = self.activation(np.dot(self.Ws[i], a) - self.bs[i])

        return a