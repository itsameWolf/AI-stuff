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
    w1 = weigth[0]
    w2 = weigth[1]
    th = weigth[2]
    y = (-(th/w2)/(th/w1))*x+(-th/w2)
    plt.plot(x,y,marker)

filename = 'iris.data'
data = np.loadtxt(filename,delimiter=',',converters={4:IrisToNumber})           #load the iris dataset

sepal_width = data[:,1]
sepal_length = data[:,0]
petal_width = data[:,3]
petal_length = data[:,2]
iris_class = data[:,4]

petal_dataset = np.column_stack((petal_width,petal_length,iris_class))
sepal_dataset = np.column_stack((sepal_width,sepal_length,iris_class))

petal_weight = [-4.0,-1.5,5.0]
petal_result = nn.testPerceptron(petal_weight,petal_dataset)

sepal_weigths = [3.9,-5.1,15.0]
sepal_result = nn.testPerceptron(sepal_weigths,sepal_dataset)

print('petal perceptron ', petal_result)
print('sepal perceptron score: ', sepal_result)

sepal_length_setosa = data[0:49,0]
sepal_width_setosa = data[0:49,1]

sepal_length_other = data[50:149,0]
sepal_width_other = data[50:149,1]

petal_length_setosa = data[0:49,2]
petal_width_setosa = data[0:49,3]

petal_length_other = data[50:149,2]
petal_width_other = data[50:149,3]



graphPerceptron (petal_weight,range(2),'b-')
graphPerceptron (sepal_weigths,range(6),'r-')

plt.scatter(x=sepal_width_setosa, y=sepal_length_setosa, c='r')
plt.scatter(x=sepal_width_other, y=sepal_length_other,c='r',marker='v')

plt.scatter(x=petal_width_setosa, y=petal_length_setosa, c='b')
plt.scatter(x=petal_width_other, y=petal_length_other,c='b', marker='v')

plt.xlabel('petal_width')
plt.ylabel('petal_length')
plt.title('petal_linear_Classifier')
plt.show()