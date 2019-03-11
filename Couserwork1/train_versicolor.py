import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import Perceptron as nn

def versicolor (iris):            #function used to convert the iris classification string into numerical values
    if iris == b'Iris-versicolor':
        return 1
    else:
        return 0

filename = 'iris.data'                                                  #load the iris dataset
data = np.loadtxt(filename,delimiter=',',converters={4:versicolor})
np.random.shuffle(data)                                                 #shuffle the rows of the datset

epochs = 30
learning_rate = 0.2
target_accuracy = 78

(weigths, accuracy, accuracy_progression) = nn.perceptronLearning(data,epochs,learning_rate, target_accuracy)     #run the peceptron learning algorithm

print ('accuracy outside ', nn.testPerceptron(weigths,data))

(tp,tn,fp,fn) = nn.confusionMatrix(weigths,data)

print('weigths: ', weigths)
print('max accuracy: ', max(accuracy_progression))
print('final accuracy: ', accuracy)

print('true positive: ',tp,'true negative: ',tn)
print('false positive: ',fp,'false negative: ',fn)

title = "%d_iterations_lambda=%f" %(len(accuracy_progression),learning_rate)
path = "./Plots/%s.png" %(title)

plt.title(title)
plt.ylabel('accuracy (%)')
plt.xlabel('iteration')
plt.plot(accuracy_progression)
plt.show()