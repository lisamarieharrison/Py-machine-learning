import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

data = np.genfromtxt('C:/Users/Lisa/Documents/code/machine learning/ex2/ex2data1.txt', delimiter=',')

x1 = data[:, 0]
x2 = data[:, 1]
X = np.column_stack((np.ones(x1.size), x1, x2))
y = data[:, 2]
m = x1.size

#Figure 1: Scatter plot of training data
df = pd.DataFrame(dict(x=x1, y=x2, label=y))
groups = df.groupby('label')

marker = {0: 'o', 1: '+'}
point_col = {0: 'yellow', 1: 'black'}

fig, ax = plt.subplots()
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker=marker[name], linestyle='', ms=12, mew=2, label=name, color=point_col[name])
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
ax.legend(numpoints=1, labels=["not admitted", "admitted"])

def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g[0]


iterations = 400
alpha = 0.001
#theta = np.zeros(3)
#theta.shape = (3, 1)

#gradient descent
J = 0
dJ = 10

def f(theta):
    theta_mat = np.array([theta[0], theta[1], theta[2]])
    theta_mat.shape = (3, 1)
    return 1/float(m)*sum(-y*np.log(sigmoid(np.dot(theta_mat.T, X.T))) - (1 - y)*np.log(1 - sigmoid(np.dot(theta_mat.T, X.T))))

result = optimize.minimize(f, [0, 0, 0], method='Nelder-Mead')
#print(result)

theta = result.x

x1_plot = np.linspace(30, 100, 80)
x2_plot = (0.5 - theta[0] - theta[1]*x1_plot)/theta[2]

plt.plot(x1_plot, x2_plot)

plt.show()