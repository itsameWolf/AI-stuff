import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import Perceptron as nn

def setosa (iris):              #function used to conver the iris classification string into numerical values
    if iris == b'Iris-setosa':
        return 1
    else:
        return 0

def versicolor (iris):            #function used to conver the iris classification string into numerical values
    if iris == b'Iris-versicolor':
        return 1
    else:
        return 0

def virginica (iris):            #function used to conver the iris classification string into numerical values
    if iris == b'Iris-virginica':
        return 1
    else:
        return 0

filename = 'iris.data'
data = np.loadtxt(filename,delimiter=',',converters={4:virginica})

print (nn.perceptronLearning(data,1000))