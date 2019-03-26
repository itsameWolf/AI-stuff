import numpy as np
from matplotlib import pyplot as plt

filename = "E:\Documents\GitHub\AI-stuff\Coursework2\chase_3k"

fitness = np.loadtxt(filename, usecols=(1,4,7))

prey = fitness[1::2]
predator = fitness[::2]

f, ga = plt.subplots(2,1,True,)
f.suptitle('Chase Algorithm Simulation')
ga[0].scatter(prey[:,0],prey[:,2])
ga[1].scatter(predator[:,0],predator[:,2])

plt.show()
print(prey.shape)
print(prey)