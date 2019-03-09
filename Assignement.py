import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import Perceptron as nn

def setosa (iris):            #function used to conver the iris classification string into numerical values
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

def graphPerceptron (weigth, range_x, marker):          #graph a linear classifier perceptron i.e. 2 inputs 1 output
    x = np.array(range_x)
    w1 = weigth[0]
    w2 = weigth[1]
    th = weigth[2]
    y = (-(th/w2)/(th/w1))*x+(-th/w2)
    plt.plot(x,y,marker)

filename = 'iris.data'
#data_setosa = np.loadtxt(filename,delimiter=',',converters={4:setosa})           #load the iris dataset for setosa classification
#data_versicolor = np.loadtxt(filename,delimiter=',',converters={4:versicolor})   #load the iris dataset for versicolor classifification
data_virginica = np.loadtxt(filename,delimiter=',',converters={4:virginica})     #load the iris dataset for virginica classification
print (data_virginica)
#np.random.shuffle(data_setosa)                                                    #randomly shuffle the rows of the dataset
#np.random.shuffle(data_versicolor)
np.random.shuffle(data_virginica)

#all_weigths_setosa = nn.trainPerceptron(data_setosa[:,0:4],data_setosa[:,4],return_all_weigths=1)
#all_weigths_versicolor = nn.trainPerceptron(data_versicolor[:,0:4],data_versicolor[:,4],return_all_weigths=1)
all_weigths_virginica = nn.trainPerceptron(data_virginica[:,0:4],data_virginica[:,4],return_all_weigths=1)

all_weigths = all_weigths_virginica

#test = nn.testPerceptron(all_weigths[149],data_versicolor)

#accuracy_over_itearation = nn.checkLearningProgress(all_weigths_setosa,data_setosa)
#accuracy_over_itearation = nn.checkLearningProgress(all_weigths_versicolor,data_versicolor)
#accuracy_over_itearation = nn.checkLearningProgress(all_weigths_virginica,data_virginica)

#print(accuracy_over_itearation)

#print(all_weigths)
w0 = all_weigths[:,0]
w1 = all_weigths[:,1]
w2 = all_weigths[:,2]
w3 = all_weigths[:,3]
th = all_weigths[:,4]

plt.subplot(2,1,1)
#plt.plot(np.arange(150),w0,'r',w1,'b',w2,'g',w3,'y',th,'k')
w0_entry = mpatches.Patch(color='r', label='w0')
w1_entry = mpatches.Patch(color='b', label='w1')
w2_entry = mpatches.Patch(color='g', label='w2')
w3_entry = mpatches.Patch(color='y', label='w3')
th_entry = mpatches.Patch(color='k', label='th')
plt.title('weights_change_and_perceptron_accuracy_over_iterations')
plt.ylabel('weights')
plt.legend(handles=[w0_entry,w1_entry,w2_entry,w3_entry,th_entry])

plt.subplot(2,1,2)
#plt.plot(np.arange(150),accuracy_over_itearation)
plt.xlabel('iteration')
plt.ylabel('accuracy')

#test = nn.testPerceptron(nn.trainPerceptron(data[:,0:4],data[:,4],0.2),data)

#sepal_length_setosa = data[0:49,0]
#sepal_width_setosa = data[0:49,1]

#sepal_length_other = data[50:149,0]
#sepal_width_other = data[50:149,1]

#petal_length_setosa = data[0:49,2]
#petal_width_setosa = data[0:49,3]

#petal_length_other = data[50:149,2]
#petal_width_other = data[50:149,3]

#c_setosa = 'r'
#c_other = 'b'

#plt.scatter(x=sepal_width_setosa, y=sepal_length_setosa, c='r')
#plt.scatter(x=sepal_width_other, y=sepal_length_other,c='r',marker='v')

#lt.scatter(x=petal_width_setosa, y=petal_length_setosa, c='b')
#plt.scatter(x=petal_width_other, y=petal_length_other,c='b', marker='v')

#plt.scatter(x=sepal_length_setosa, y=sepal_width_setosa, c=c_setosa)
#plt.scatter(x=sepal_length_other, y=sepal_width_other,c=c_other)

#plt.scatter(x=petal_length_setosa, y=petal_width_setosa, c=c_setosa, marker='v')
#plt.scatter(x=petal_length_other, y=petal_width_other,c=c_other, marker='v')

#plt.xlabel('petal_width')
#plt.ylabel('petal_length')
#plt.title('petal_linear_Classifier')

plt.show()

input()