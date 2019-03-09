import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns

filename = 'iris.data'

sns.set(style="ticks") 

iris = pd.read_csv(filename,header=None,names=['sepal_length','sepal_width','petal_length','petal_width','species'])

sns.pairplot(iris, hue="species")

plt.show()