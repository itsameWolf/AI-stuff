import numpy as np

def sigmoid(x):                         #Sigmoid function
    return (1.0/(1.0+2.71828**(-x)))

def solvePerceptron(weigths, inputs, activation = 0):               #given the weigths of a perceptron and the inputs calculate its output
    dotProduct = np.dot(weigths,inputs)                             #perform the dot product of inputs and outputs
    if (activation == 0):                                           #step activation function
        if (dotProduct <= 0):
            return 0
        else:
            return 1
    elif (activation == 1):                                         #sigmoid activation function
        return sigmoid(dotProduct)
    
def trainPerceptron(inputs, outputs, starting_weigts = 0, learning_rate = 1, return_all_weigths = 0,):            #train a peceptron using a dataset
    n_weights = np.ma.size(inputs,1) +1                             #how many weights are needed for the perceptron includind the threshold
    if np.any(starting_weigts) == 0:
        weigths = np.random.rand(n_weights)                         #initialise the weigths with random values
    else:
        weigths = starting_weigts
    i_l = np.ma.size(inputs,0)                                      #lerngth of the datset
    ones = np.ones((i_l,1))                                         #create a column vector of one with as many rows as the input dataset
    inputs_ww = np.concatenate((inputs,ones),axis=1)                #add the one column vector to the input so that we can treat the threshold as a weight
    weigth_storage = np.empty((i_l,n_weights))
    
    for i in range(0,i_l):                                          #perceptron learning algorithm

        output = solvePerceptron(weigths, inputs_ww[i])
        input_i = inputs_ww[i]

        if output < outputs[i]:                                     #if the output is smaller than expected add the inputs to the respective weigths
            np.add(weigths,input_i*learning_rate,weigths)
        elif output > outputs[i]:                                   #if the output is bigger than expected subtract the inputs to the respective weigths
            np.subtract(weigths,input_i*learning_rate,weigths)
        else:
            weigths = weigths                                       #if the output is as expected keep the wigths unchanged

        weigth_storage[i] = weigths
    if return_all_weigths:
        return weigth_storage
    else:
        return weigths

def testPerceptron(weights, dataset):                               #test a perceptron over a dataset and see how many items have been classified coreectly

    i_l = np.ma.size(dataset,0)                                     #length of the dataset
    data_width = np.size(weights)-1                                 #number of columns in the dataset
    inputs = dataset[:,0:data_width]                                #separate the inputs 
    ones = np.ones((i_l,1))                     
    inputs_ww = np.concatenate((inputs,ones),axis=1)                #add a one column vector for the threshold weigth
    outputs = dataset[:,data_width]                                 #separate the outputs from the rest of the dataset

    correctly_classified = 0                                        #initialise the counter for the items that have been classified correctly

    for i in range(0,i_l):
        output = solvePerceptron(weights,inputs_ww[i],0)
        if  output == outputs[i]:                                   #whenever an item has been classified correctly increment the counter
            correctly_classified += 1
    return (correctly_classified/i_l)*100                           #return the percentage of items that have been classified correctly          

def checkLearningProgress (weight_matrix,dataset):
    i_l = np.ma.size(weight_matrix,0)
    accuracy = np.empty(i_l)
    for i in range(i_l):
        accuracy[i] = testPerceptron(weight_matrix[i],dataset)
    return accuracy

def perceptronLearning (dataset, max_iterations, learning_rate = 1):

    data_width = np.ma.size(dataset,1)
    data_length = np.ma.size(dataset,0)

    ones = np.ones((data_length,1))                                  
    inputs = dataset[:,0:data_width-1]
    inputs_ww = np.concatenate((inputs,ones),axis=1)
    outputs = dataset[:,data_width-1]

    weigths = np.random.rand(data_width)
    
    i = 0
    
    while i < max_iterations:

        for j in range(data_length):

            output = solvePerceptron(weigths, inputs_ww[j])
            input_i = inputs_ww[j]

            if output < outputs[j]:

                np.add(weigths,input_i*learning_rate,weigths)

            elif output > outputs[j]:

                np.subtract(weigths,input_i*learning_rate,weigths)

            else:

                weigths = weigths
                
        accuracy = testPerceptron(weigths, dataset)
        print(i)
        i += 1
        if accuracy == 100:
            break
    
    return weigths, accuracy