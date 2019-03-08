import numpy as np

def sigmoid(x):
    return (1.0/(1.0+2.71828**(-x)))

def solvePerceptron(weigths, inputs, activation):
    dotProduct = np.dot(weigths,inputs)
    #print(dotProduct)
    if (activation == 0):
        if (dotProduct <= 0):
            return 0
        else:
            return 1
    elif (activation == 1):
        return sigmoid(dotProduct)
    
def trainPerceptron(inputs, outputs):
    n_weights = np.ma.size(inputs,1) +1
    i_l = np.ma.size(inputs,0)
    ones = np.ones((i_l,1))
    
    inputs_ww = np.concatenate((inputs,ones),axis=1)
    #print(inputs_ww)
    weigths = np.random.rand(n_weights)
    #print(weigths)
    for i in range(0,i_l):
        output = solvePerceptron(weigths, inputs_ww[i], 0)
        #print('perceptron output ',output)
        input_i = inputs_ww[i]
        if outputs[i] == 1:
            if output == 0:
                np.add(weigths,input_i,weigths)

        elif outputs[i] == 0:
            if output == 1:
                np.subtract(weigths,input_i,weigths)
        else:
            weigths = weigths        
    return weigths

def testPerceptron(weights, dataset):
    i_l = np.ma.size(dataset,0)
    data_width = np.size(weights)-1
    inputs = dataset[:,0:data_width]
    ones = np.ones((i_l,1))
    inputs_ww = np.concatenate((inputs,ones),axis=1)
    outputs = dataset[:,data_width]

    correctly_classified = 0

    for i in range(0,i_l):
        if solvePerceptron(weights,inputs_ww[i],0) == outputs[i]:
            correctly_classified += 1
    return correctly_classified/i_l