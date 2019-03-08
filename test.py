import csv
import numpy as np
import matplotlib.pyplot as plt
import Perceptron as nn

def graphPerceptron (weigth, range_x, marker):
    x = np.array(range_x)
    y = ((weigth[0]*x)+weigth[2])/weigth[1]
    plt.plot(x,y,marker)


graphPerceptron ([1.0,1.0,1.0],range(10),'g-')

plt.show()