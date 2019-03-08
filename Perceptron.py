def sigmoid(x):
    return (1.0/(1.0+2.71828**(-x)))

def solvePerceptron(weigths, inputs, threshold, activation):
    dotProduct = 0
    inputs.append(-1)
    weigths.append(threshold)
    for i in range(0,len(weigths)):
        dotProduct = weigths[i] * inputs[i] + dotProduct
    if (activation == 0):
        if (dotProduct <= 0):
            return 0
        else:
            return 1
    elif (activation == 1):
        return sigmoid(dotProduct)
    

