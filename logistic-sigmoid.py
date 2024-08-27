import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    sigmoid_g = 1/(1+np.exp(-z))
    return sigmoid_g

z_tmp = np.arange(-10,11)
y_tmp = sigmoid(z_tmp)

def plot_show(z_tmp,y_tmp):
    plt.scatter(z_tmp, y_tmp, color='red', label='Actual')
    plt.plot(z_tmp,y_tmp,c='b')
    plt.xlabel("z")
    plt.ylabel("y")
    plt.show()

plot_show(z_tmp,y_tmp)

def logistic(x,w,b):
    g = np.zeros(x.shape[0])
    for i in range(x.shape[0]):    
        z = np.dot(w,x[i])+b
        g[i] = 1/(1+np.exp(-z))
    return g

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
w = np.zeros((1))
b = 0

g_z = logistic(x_train,w,b)
print(g_z)
plot_show(g_z,y_train)
