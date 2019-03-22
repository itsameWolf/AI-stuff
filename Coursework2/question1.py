import numpy as np
from matplotlib import pyplot as plt

filename = "E:\Documents\GitHub\AI-stuff\Coursework2\mice_simulation_one.txt"

fitness = np.loadtxt(filename, usecols=(1,4,7))

data = fitness[0:1000]

plt.scatter(data[:,0],data[:,2])
plt.show()
#print(fitness.shape)
#print(fitness[1])