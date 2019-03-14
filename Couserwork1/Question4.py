import numpy as np
import Perceptron as nn

def iris (iris):            #function used to conver the iris classification string into numerical values
    if iris == b'Iris-setosa':
        return 1
    elif iris == b'Iris-versicolor':
        return 2
    elif iris == b'Iris-virginica':
        return 3


def irisClassifier1(sepal_length, sepal_width, petal_length, petal_width):                          #classifier using 3 perceptrons in the hidden layer

    inputs = [sepal_length, sepal_width, petal_length, petal_width, 1]                              #input layer

    weight_input_setosa = [-0.50070756,  2.35675675, -1.57468868, -1.05504908,  0.02127709]         #hidden layer perceptrns
    weight_input_versicolor = [ 2.96015761, -5.7989648,   1.04779707, -5.10024532,  1.98070432]
    weight_input_virginica = [-2.50485896, -2.20186858,  3.62850465,  3.19246628, -0.98835995]

    setosa_classifier = nn.solvePerceptron(weight_input_setosa,inputs)                              #hidden layer outputs
    versicolor_classifier = nn.solvePerceptron(weight_input_versicolor,inputs)
    virginica_classifier = nn.solvePerceptron(weight_input_virginica,inputs)

    middle_layer = [setosa_classifier, versicolor_classifier, virginica_classifier, 1]

    weight_output_setosa = [ 1.0, 0.0, 0.0, -0.5]                                                   #output layer perceptrons
    weight_output_versicolor = [-0.5, 1, -0.5, -0.6]
    weight_output_virginica = [-1.1, -0.6, 0.5, 0.5]

    setosa_output = nn.solvePerceptron(weight_output_setosa,middle_layer)                           #outputs
    versicolor_output = nn.solvePerceptron(weight_output_versicolor, middle_layer)
    virginica_output = nn.solvePerceptron(weight_output_virginica,middle_layer)

    outputs = [setosa_output, versicolor_output, virginica_output]

    return outputs

def irisClassifier2(sepal_length, sepal_width, petal_length, petal_width):                          #classifier using 3 perceptrons in the hidden layer

    inputs = [sepal_length, sepal_width, petal_length, petal_width, 1]                              #input layer

    weight_input_setosa = [-0.50070756,  2.35675675, -1.57468868, -1.05504908,  0.02127709]         #hidden layer perceptrons
    weight_input_virginica = [-2.50485896, -2.20186858,  3.62850465,  3.19246628, -0.98835995]

    setosa_classifier = nn.solvePerceptron(weight_input_setosa,inputs)                              #hidden layer outputs
    virginica_classifier = nn.solvePerceptron(weight_input_virginica,inputs)

    middle_layer = [setosa_classifier, virginica_classifier, 1]

    weight_output_setosa = [ 1.0, 0.0, -0.5]                                                        #output layer perceptrons
    weight_output_versicolor = [-0.5, -0.5, 0.4]
    weight_output_virginica = [-0.6, 0.6, -0.5,]

    setosa_output = nn.solvePerceptron(weight_output_setosa,middle_layer)                           #outputs
    versicolor_output = nn.solvePerceptron(weight_output_versicolor, middle_layer)
    virginica_output = nn.solvePerceptron(weight_output_virginica,middle_layer)

    outputs = [setosa_output, versicolor_output, virginica_output]

    return outputs

filename = 'iris.data'

data = np.loadtxt(filename,delimiter=',',converters={4:iris})     #load the iris dataset for virginica classification

iris_input = data[:,0:4]

i = 1

while i:
    sl = float(input('Sepal length: '))
    sw = float(input('Sepal width: '))
    pl = float(input('Petal length: ')) 
    pw = float(input('Petal width: '))

    classifier = int(input('which classifier? 3 hidden perceptron or 2 hidden perceptron (1/2): '))

    if classifier == 1:
        outputs = irisClassifier1(sl, sw, pl, pw)
    elif classifier == 2:
        outputs = irisClassifier2(sl, sw, pl, pw)

    print('Result: ', outputs)

    stop = input('do you want to classify another iris? (y/n): ')

    if stop == 'n':
        i = 0

#for i in range(0, np.ma.size(iris_input,0)):

<<<<<<< HEAD:Question4.py
#    print(i, irisClassifier1(iris_input[i,0],iris_input[i,1],iris_input[i,2],iris_input[i,3]))

#for i in range(0, np.ma.size(iris_input,0)):

#    print(i, irisClassifier2(iris_input[i,0],iris_input[i,1],iris_input[i,2],iris_input[i,3]))
=======
#    print(i, irisClassifier(iris_input[i,0],iris_input[i,1],iris_input[i,2],iris_input[i,3]))

#for i in range(0, np.ma.size(iris_input,0)):

#    print(i, irisClassifier(iris_input[i,0],iris_input[i,1],iris_input[i,2],iris_input[i,3]))
>>>>>>> 642768d4038b66ba385ecc70aad9e8c6cf60cf22:Couserwork1/Question4.py
