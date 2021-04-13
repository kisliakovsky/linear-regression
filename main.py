import numpy
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import seaborn
from seaborn import FacetGrid
from pandas import DataFrame

X = numpy.array([[-10], [-7], [-5], [1], [9]])
y = numpy.array([-5, -5, -4, -1, 4])
reg = LinearRegression().fit(X, y)
x_plot = numpy.arange(-10, 12, 2)
y_plot = reg.predict(numpy.array([x_plot]).transpose())

seaborn.set_style('whitegrid')
grid = FacetGrid(DataFrame({'x': X.transpose()[0], 'y': y}), height=6)
grid.map(pyplot.scatter, 'x', 'y', edgecolor='w')
plot = pyplot.plot(x_plot, y_plot, color='r')
pyplot.savefig('output.png')
