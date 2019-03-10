import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import Perceptron as nn

sl = 0
sw = 0
pl = 0 
pw = 0

inputs = [sl, sw, pl, pw, 1]

weight_input_setosa = [-0.50070756,  2.35675675, -1.57468868, -1.05504908,  0.02127709]
weight_input_versicolor = [ 2.96015761, -5.7989648,   1.04779707, -5.10024532,  1.98070432]
weight_input_virginica = [-2.50485896, -2.20186858,  3.62850465,  3.19246628, -0.98835995]

weight_output_setosa = [ 1.0, 0.0, 0.0, 0.5]
weight_output_versicolor = [-0.6, 1, -0.6, -0.6]
weight_output_virginica = [-1.1, -0.6, 0.5, 0.5]

setosa_classifier = nn.solvePerceptron(weight_input_setosa,inputs)
versicolor_classifier = nn.solvePerceptron(weight_input_versicolor,inputs)
virginica_classifier = nn.solvePerceptron(weight_input_virginica,inputs)

middle_layer = [setosa_classifier, versicolor_classifier, virginica_classifier, 1]

setosa_output = nn.solvePerceptron(weight_output_setosa,middle_layer)
versicolor_output = nn.solvePerceptron(weight_output_versicolor, middle_layer)
virginica_output = nn.solvePerceptron(weight_output_virginica,middle_layer)

setosa_output = [setosa_output, versicolor_output, virginica_output]