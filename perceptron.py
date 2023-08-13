import math
import numpy as np
import matplotlib.pyplot as plt

# A simple perceptron class that creates a weight vector with small random numbers when initialized
# Has methods to compute the activation for an input and to produce a binary class label for an input
# Doesn't include a bias term.
class Perceptron:
    def __init__(self, n):
        """
        n (int) - The length of x vectors that the classifier will process.
        """
        
        # Start with weights uniformly chosen from the range [-0.5, 0.5]
        self.weights = (np.random.rand(n) - 0.5).tolist()
        
    def activation(self, x):
        return np.dot(x, self.weights)

    
    def predict(self, x):
        return 1 if self.activation(x) > 0 else -1
    
# Compute hinge loss
def loss_hinge(y, activation):
    # If class label and activation have same sign, hinge loss is 0; otherwise it is -y*activation
    if(y > 0 and activation > 0):
        return 0
    elif(y < 0 and activation < 0):
        return 0
    else:
        return -y * activation

# Code to check hinge loss implementation
for y in [-1, 1]:
    activation = np.arange(-5, 5, 0.1)
    plt.plot(activation, [loss_hinge(y, a) for a in activation])
    plt.title('y = %d' % y)
    plt.xlabel('activation')
    plt.ylabel('loss')
    plt.show()

# Method of Finite Differences
def finite_dif(clf,x,y,weights,index,epsilon,loss_fn):
    L1 = weights.copy()
    L2 = weights.copy()
    
    L1[index] += epsilon
    L2[index] -= epsilon
    
    # get activation (w2x2 + w1x1 + w0..)
    L1 = np.dot(x, L1)
    L2 = np.dot(x, L2)
    
    # find loss
    L1 = loss_fn(y,L1)
    L2 = loss_fn(y,L2)
    
    return((L1-L2)/(2*epsilon))

def gd_step(clf, x, y, learning_rate, loss_fn, epsilon = 0.001):
    newWeights = []
    for i in range(len(clf.weights)):
        fd = finite_dif(clf,x,y,clf.weights,i,epsilon,loss_fn)
        newWeights.append(clf.weights[i] - learning_rate*fd)
    return newWeights

# Test cases
clf = Perceptron(1)
clf.weights = [2]

print(gd_step(clf, [1], 1, 0.1, loss_hinge, epsilon = 0.001))
print(gd_step(clf, [1], -1, 0.1, loss_hinge, epsilon = 0.001))
print(["{0:0.2f}".format(i) for i in gd_step(clf, [1], -1, 0.1, lambda a, b: (b - 1) * (b - 1), epsilon = 0.001)])

clf = Perceptron(10)
clf.weights = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
print(["{0:0.2f}".format(i) for i in gd_step(clf, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], -1, 0.1, loss_hinge, epsilon = 0.001)])



# Training Data
# Generate a simple 2D dataset of n positive examples followed by n negative examples.
# Prepends a 1 in each example so that the bias term corresponds to the first weight
n = 10
X = np.concatenate((np.random.rand(n, 2) + 1,
                    np.random.rand(n, 2)))
X = np.hstack((np.expand_dims(np.ones(2*n), 1), X))
Y = [1] * n + [-1] * n
colors = c = ['r'] * n + ['g'] * n

# Randomize the order of the instances just for fun
rng = np.random.default_rng()
state = rng.__getstate__()
rng.shuffle(X)
rng.__setstate__(state)
rng.shuffle(Y)
rng.__setstate__(state)
rng.shuffle(colors)

# Plots the dataset
plt.scatter(X[:,1], X[:,2], c = colors)


# Full Gradient Descent
# Run until one epoch produces no classification errors
def fullGD(clf,task):
    done = False
    while not done:
        done = True
        for x, y in zip(X, Y):
            if clf.predict(x) * y <= 0: # if prediction is wrong
                done = False
            if(task == 4):
                clf.weights = gd_step(clf, x, y, 0.01, loss_hinge)
            else:
                clf.weights = gd_step(clf, x, y, 0.01, loss_hinge_margin)


# Plotting Hyperplanes
# w0 + w1x1 + w2x2 = 0 means we're on the line
# have the function solve for x2 and plug x2 into the form below:

# x2 = mx1 + b

def weights_to_slope_intercept(weights):
    # Extract the weight parameters
    w0, w1, w2 = weights[0], weights[1], weights[2]
    
    # Compute the slope and intercept
    slope = -w1 / w2
    intercept = -w0 / w2
    
    return slope, intercept

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

plt.xlim([0, 2])
plt.ylim([0, 2])
plt.scatter(X[:,1], X[:,2], c = colors)
abline(-1, 2)
abline(2, 0)

# Green Line
clf1 = Perceptron(3)
fullGD(clf1,4)
m,b = weights_to_slope_intercept(clf1.weights)
abline(m,b)

# Red Line
clf2 = Perceptron(3)
fullGD(clf2,4)
m,b = weights_to_slope_intercept(clf2.weights)
abline(m,b)

# Purple Line
clf3 = Perceptron(3)
fullGD(clf3,4)
m,b = weights_to_slope_intercept(clf3.weights)
abline(m,b)

# Brown Line
clf4 = Perceptron(3)
fullGD(clf4,4)
m,b = weights_to_slope_intercept(clf4.weights)
abline(m,b)

# Pink Line
clf5 = Perceptron(3)
fullGD(clf5,4)
m,b = weights_to_slope_intercept(clf5.weights)
abline(m,b)

# I notice that all of the lines created do a very good job at separating the data into two sides.
# Additionally, there seems to be alot of variation between different runs of the perceptron 
# algorithm, but all are separating the data correctly, indicating that the algorithm is 
# works as intended but isn't necessarily consistent.


# Try out a new loss function
def loss_hinge_margin(y, activation):
    return 0 if y * activation > 1 else -y * activation + 1
# Plot the data
n = 10
X = np.concatenate((np.random.rand(n, 2) + 1,
                    np.random.rand(n, 2)))
X = np.hstack((np.expand_dims(np.ones(2*n), 1), X))
Y = [1] * n + [-1] * n
colors = c = ['r'] * n + ['g'] * n

# Randomize the order of the instances just for fun
rng = np.random.default_rng()
state = rng.__getstate__()
rng.shuffle(X)
rng.__setstate__(state)
rng.shuffle(Y)
rng.__setstate__(state)
rng.shuffle(colors)

plt.xlim([0, 2])
plt.ylim([0, 2])
plt.scatter(X[:,1], X[:,2], c = colors)
abline(-1, 2)
abline(2, 0)

# Green Line
clf6 = Perceptron(3)
fullGD(clf6,5)
m,b = weights_to_slope_intercept(clf6.weights)
abline(m,b)

# Red Line
clf7 = Perceptron(3)
fullGD(clf7,5)
m,b = weights_to_slope_intercept(clf7.weights)
abline(m,b)

# Purple Line
clf8 = Perceptron(3)
fullGD(clf8,5)
m,b = weights_to_slope_intercept(clf8.weights)
abline(m,b)

# Brown Line
clf9 = Perceptron(3)
fullGD(clf9,5)
m,b = weights_to_slope_intercept(clf9.weights)
abline(m,b)

# Pink Line
clf10 = Perceptron(3)
fullGD(clf10,5)
m,b = weights_to_slope_intercept(clf10.weights)
abline(m,b)


# Using this new loss function, it seems that the plotted lines are more consistent with eachother.
# Before, there was a lot of variation in the angle the line came at, desipte it still separating the data.
# The results look the way they do because the new loss function increases the loss calculated by gradient descent.
# This causes the weights to move more drastically and results in the lines generally falling into the same general
# angle and matching up with other linear separators. This does come at a cost, as while theres less variance
# in the linear separators, they tend to be towards one side of the data, not in between. As a line of best fit, it
# gives up some accuracy in return for consistency.