# Online Learning

# This file Solves the Linear Regression Problem using
# A - Closed Form Solution which uses Pseudo Inverse W = pseudoInverse(X)*Y
# B - Gradient Descent Method
# Linear Regression problem: Find W which minimizes (Y - WX)^2
# Data required for Linear Regression is generated from different file (GenerateData.py)

# Import Required Libraries
import numpy
import matplotlib.pyplot as plt

# Functions
# Calculates Gradient of the given Function
def Gradient(w0, w1, x, y):
    g1 = (2 * w0) + (2 * w1 * x) - (2 * y)
    g2 = (2 * w0 * x) + (2 * w1 * x**2) - (2 * y * x)
    return numpy.array([g1, g2])

# Update the points based on the Gradient Descent Algorithm
def Update_Weights(x, y, w0, w1, eta):
    return numpy.array([w0, w1]) - eta*Gradient(w0, w1, x, y)

# Actual Loop where Gradient Descent Algo runs until optimal point is reached
def GD(X, Y, max_iter, tol, eta):
    iterations = 0
    Epoch = numpy.array([])
    row, col = X.shape
    W_Initial_Guess = numpy.loadtxt('W.txt')
    while iterations < max_iter:
        if iterations == 0:
            w_temp = W_Initial_Guess
            W_GD = numpy.array([W_Initial_Guess])
            for i in range(0, row):
                w_temp = Update_Weights(X[i, 1], Y[i], w_temp[0], w_temp[1], eta)
            W_GD = numpy.concatenate((W_GD, [w_temp]), axis=0)
            Epoch = numpy.concatenate((Epoch, [iterations]), axis=0)
            print('No. of Iterations: ', iterations, ' Linear Least Square Fit: ', W_GD[-1], '\n')
            iterations += 1
        else:
            # Run The Gradient Descent Algorithm
            for i in range(0, row):
                w_temp = Update_Weights(X[i, 1], Y[i], w_temp[0], w_temp[1], eta)
            # Book Keeping
            W_GD = numpy.concatenate((W_GD, [w_temp]), axis=0)
            Epoch = numpy.concatenate((Epoch, [iterations]), axis=0)
            print('No. of Iterations: ', iterations, ' Linear Least Square Fit: ', W_GD[-1], '\n')
            # Check for Close Weights
            if (W_GD[-1] - W_GD[-2]).all() < tol:
                print('Optimal Value Reached')
                break
            else:
                iterations += 1
    return Epoch, W_GD

# Function for plotting
def grapher(W):
    x = numpy.array([0, 51])
    y = numpy.array([W[0], W[0] + 51 * W[1]])
    return x, y

# Parameters
# Shape of matrix X = Nx2
# Shape of matrix Y = Nx1
# Shape of Weight matrix = 2x1
X = numpy.loadtxt('X.txt')
Y = numpy.loadtxt('Y.txt')
max_iter = 100000 # Maximum Iterations to reach the optimal value
tol = 1.0e-6 # Tolerance
eta = 0.00002 # Learning Rate

# Solution for the Linear Regression Problem using Closed Form Solution
# From the way I have chosen matrices the
# Equation will be: Y - X * W = 0
# Hence closed form solution will be W = pseudoInverse(X)*Y
# Pseudo Inverse also known as Penrose Inverse
W_Closed_Form_Solution = numpy.dot(numpy.linalg.pinv(X), Y)
numpy.savetxt('ClosedFormSolution.txt', W_Closed_Form_Solution)


# Solution for the Linear Regression Problem using Gradient Descent Method
if numpy.DataSource().exists('OnlineLearningGDSolution.txt'):
    W_GD = numpy.loadtxt('OnlineLearningGDSolution.txt')  # Load the Initial Weights
else:
    Epoch, W_GD1 = GD(X, Y, max_iter, tol, eta)
    W_GD = W_GD1[-1]
    numpy.savetxt('OnlineLearningGDSolution.txt', W_GD1[-1])  # Generate the Weights and save them

# Final Optimal Point
print(W_GD)
print(W_Closed_Form_Solution)

# Plot the results
# Plot 1
fig, ax1 = plt.subplots()
temp_x, temp_y = grapher(W_Closed_Form_Solution)
ax1.plot(temp_x, temp_y, 'r--')
ax1.plot(X[:, 1], Y)
plt.title('Closed Form Solution for Linear Least Square Fit')
plt.xlabel('X')
plt.ylabel('Y')
# Plot 2
fig, ax2 = plt.subplots()
temp_x, temp_y = grapher(W_GD)
ax2.plot(temp_x, temp_y, 'r--')
ax2.plot(X[:, 1], Y)
plt.title('Online Gradient Descent for Linear Least Square Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()