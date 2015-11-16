import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt('C:/Users/Lisa/Documents/code/machine learning/ex1/ex1data1.txt', delimiter=',')

x = data[:,0]
y = data[:,1]
m = x.size

#Figure 1: Scatter plot of training data
plt.plot(x, y, 'o')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
#plt.show()

X = np.column_stack((np.ones(m), x))  # add column of 1s for the intercept
theta = np.zeros(2)

iterations = 1500
alpha = 0.01

#gradient descent
J = 10000
dJ = 10000
while abs(dJ) > 10e-10:

    h = theta[0] + theta[1]*x
    theta[0] -= alpha*(1/float(m))*sum(h-y)
    theta[1] -= alpha*(1/float(m))*sum((h-y)*x)

    dJ = J - sum(theta)
    J = sum(theta)

line_best_fit = theta[0] + theta[1]*x

plt.plot(x, line_best_fit)
#plt.show()

#plot cost function
size = 50
theta_0 = np.repeat(np.linspace(-10, 10, size), size)
theta_1 = np.tile(np.linspace(-1, 4, size), size)
J = [0]*theta_0.size
for l in range(0, len(theta_0)):
    J[l] = 1/(2*float(m))*sum((theta_0[l] + theta_1[l]*x - y)**2)

'''
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.linspace(-10, 10, size)
Y = np.linspace(-1, 4, size)
X, Y = np.meshgrid(Y, X)
Z = np.array(J).reshape(X.shape)

surf = ax.plot_surface(Y, X, Z, rstride=1, cstride=1, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
ax.set_zlim(0, 800)

plt.show()

'''

#normal equation for comparison
print np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
print theta









