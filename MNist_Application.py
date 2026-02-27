import numpy as np
import random as rm
import matplotlib.pyplot as plt

class MNist():
    
    def __init__(self):

        self.ims_train = np.load('images_train.npy')
        self.lbs_train = np.load('labels_train.npy')
        self.ims_test = np.load('images_test.npy')
        self.lbs_test = np.load('labels_test.npy')

def main():
    mn = MNist()

    test = True

    if test:
    
        i = rm.randint(0, len(mn.ims_train))
        j = rm.randint(0, len(mn.ims_test))

        plt.imshow(mn.ims_train[i], cmap='gray')
        plt.title("Number: "+str((mn.lbs_train[i])))
        plt.show()

        plt.imshow(mn.ims_test[j], cmap='gray')
        plt.title("Number: "+str((mn.lbs_test[j])))
        plt.show()


if __name__ == "__main__":
    main()