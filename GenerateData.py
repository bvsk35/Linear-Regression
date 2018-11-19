# This file generates data required for Linear Regression

# Import required libraries
import numpy

# To generate X
a = numpy.arange(1, 51)
b = numpy.ones(50)
X = numpy.concatenate((b[:, numpy.newaxis], a[:, numpy.newaxis]), axis=1)
numpy.savetxt('X.txt', X)

# To generate Y
A = numpy.arange(1, 51)
B = numpy.random.uniform(-1, 1, (50))
Y = A+B
numpy.savetxt('Y.txt', Y)

# Inital weights for Y = W0 + W1 * X
W = numpy.random.uniform(-1, 1, (1, 2))
numpy.savetxt('W.txt', W)