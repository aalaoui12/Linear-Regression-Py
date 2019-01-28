import numpy as np  # numpy is for scientific calculation
import pandas as pd  # pandas helps with data analysis
import matplotlib.pyplot as plt


data = pd.read_csv("machine-learning-ex/ex1/ex1data1.txt")
X = data.iloc[:, 0]  # set x to first column, y to second column
y = data.iloc[:, 1]
m = len(y)
data.head()

plt.scatter(X, y)
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profits ($10,000s)")

X = X[:, np.newaxis]  # increases number of columns by 1
y = y[:, np.newaxis]  # always do this for both X and y
theta = np.zeros([2, 1])  # just like matlab
iterations = 1500
alpha = .01
ones = np.ones((m, 1))
X = np.hstack((ones, X))  # adds ones to first column of X


def compute_Cost(X, y, theta):
    # note that numpy arrays, like below, can be directly subtracted
    product = np.dot(X, theta) - y  # matrix multiplication of X and theta
    return np.sum(np.power(temp, 2)) / (2*m)


J = compute_Cost(X, y, theta)
print(J)


def gradient_Descent(X, y, theta, alpha, iterations):
    for x in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta


theta = gradient_Descent(X, y, theta, alpha, iterations)
print(theta)

print(compute_Cost(X, y, theta))  # J is minimized
