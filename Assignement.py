import numpy as np
import matplotlib.pyplot as plt
import Perceptron as nn

def IrisToNumber (iris):            #function used to conver the iris classification string into numerical values
    if iris == b'Iris-setosa':
        return 1
    else:
        return 0

def graphPerceptron (weigth, range_x, marker):          #graph a linear classifier perceptron i.e. 2 inputs 1 output
    x = np.array(range_x)
    y = ((weigth[0]*x)/weigth[1])+(weigth[2])/weigth[1]
    plt.plot(x,y,marker)

filename = 'iris.data'
data = np.loadtxt(filename,delimiter=',',converters={4:IrisToNumber})           #load the iris dataset
np.random.shuffle(data)                                                         #randomly shuffle the rows of the dataset

test = nn.testPerceptron(nn.trainPerceptron(data[:,0:4],data[:,4],0.2),data)

print(test)

#final_weigths1 = nn.trainPerceptron(data[:,0:2],data[:,4])
#final_weigths2 = nn.trainPerceptron(data[:,2:4],data[:,4])

#print(final_weigths1)
#print(final_weigths2)

#graphPerceptron (final_weigths1,range(10),'g-')
#graphPerceptron (final_weigths2,range(10),'y-')

sepal_length_setosa = data[0:49,0]
sepal_width_setosa = data[0:49,1]

sepal_length_other = data[50:149,0]
sepal_width_other = data[50:149,1]

petal_length_setosa = data[0:49,2]
#petal_width_setosa = data[0:49,3]

petal_length_other = data[50:149,2]
petal_width_other = data[50:149,3]

c_setosa = 'r'
c_other = 'b'

#plt.scatter(x=sepal_width_setosa, y=sepal_length_setosa, c=c_setosa)
#plt.scatter(x=sepal_width_other, y=sepal_length_other,c=c_other)

#plt.scatter(x=petal_width_setosa, y=petal_length_setosa, c=c_setosa, marker='v')
#plt.scatter(x=petal_width_other, y=petal_length_other,c=c_other, marker='v')

#plt.scatter(x=sepal_length_setosa, y=sepal_width_setosa, c=c_setosa)
#plt.scatter(x=sepal_length_other, y=sepal_width_other,c=c_other)

#plt.scatter(x=petal_length_setosa, y=petal_width_setosa, c=c_setosa, marker='v')
#plt.scatter(x=petal_length_other, y=petal_width_other,c=c_other, marker='v')

#plt.show()