import numpy as np
import matplotlib.pyplot as plt

def IrisToNumber (iris):            #function used to conver the iris classification string into numerical values
    if iris == b'Iris-setosa':
        return 1
    else:
        return 0

filename = 'iris.data'
data = np.loadtxt(filename,delimiter=',',converters={4:IrisToNumber})           #load the iris dataset

sepal_length_setosa = data[0:49,0]
sepal_width_setosa = data[0:49,1]

sepal_length_versicolor = data[50:99,0]
sepal_width_versicolor = data[50:99,1]

sepal_length_virginica = data[100:149,0]
sepal_width_virginica = data[100:149,1]

petal_length_setosa = data[0:49,2]
petal_width_setosa = data[0:49,3]

petal_length_versicolor = data[50:99,2]
petal_width_versicolor = data[50:99,3]

petal_length_virginica = data[100:149,2]
petal_width_virginica = data[100:149,3]

c_setosa = 'r'
c_versicolor = 'b'
c_virginica = 'g'

plt.scatter(x=sepal_width_setosa, y=sepal_length_setosa, c=c_setosa)
plt.scatter(x=sepal_width_versicolor, y=sepal_length_versicolor,c=c_versicolor)
plt.scatter(x=sepal_width_virginica, y=sepal_length_virginica,c=c_virginica)

plt.scatter(x=petal_width_setosa, y=petal_length_setosa, c=c_setosa, marker='v')
plt.scatter(x=petal_width_versicolor, y=petal_length_versicolor,c=c_versicolor, marker='v')
plt.scatter(x=petal_width_virginica, y=petal_length_virginica,c=c_virginica, marker='v')

#plt.scatter(x=petal_width_setosa, y=petal_length_setosa, c=c_setosa, marker='v')
#plt.scatter(x=petal_width_other, y=petal_length_other,c=c_other, marker='v')

#plt.scatter(x=sepal_length_setosa, y=sepal_width_setosa, c=c_setosa)
#plt.scatter(x=sepal_length_other, y=sepal_width_other,c=c_other)

#plt.scatter(x=petal_length_setosa, y=petal_width_setosa, c=c_setosa, marker='v')
#plt.scatter(x=petal_length_other, y=petal_width_other,c=c_other, marker='v')

plt.show()