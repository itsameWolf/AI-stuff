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
data = np.loadtxt(filename,delimiter=',',converters={4:setosa})
np.random.shuffle(data)

iterations = 30
learning_rate = 1

(weigths, accuracy, accuracy_progression) = nn.perceptronLearning(data,iterations,learning_rate,100)

(tp,tn,fp,fn) = nn.confusionMatrix(weigths,data)

print('weigths: ', weigths)
print('accuracy: ', accuracy)

print('true positive: %d    true negative: %d',(tp,tn))
print('false positive: %d   false negative: %d',(fp,fn))

title = "%d_iterations_lambda=%f" %(len(accuracy_progression),learning_rate)
path = "./Plots/%s.png" %(title)

plt.title(title)
plt.ylabel('accuracy (%)')
plt.xlabel('iteration')
plt.plot(accuracy_progression)
plt.show()