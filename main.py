import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

def loss_function(x):
    return np.sum(x**2)

def numerical_gradient(loss_function, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        temp = x[i]

        x[i] = temp + h
        fxh1 = loss_function(x)

        x[i] = temp - h
        fxh2 = loss_function(x)

        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = temp

    return grad

def gradient_descent(loss_function, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(loss_function, x)
        x -= lr * grad

    return x

if __name__=='__main__':
    #print(numerical_gradient(loss_function, np.array([3.0, 4.0])))
    #print(numerical_gradient(loss_function, np.array([0.0, 2.0])))
    #print(numerical_gradient(loss_function, np.array([3.0, 0.0])))

    init_x = np.array([-3.0, 4.0])
    result = gradient_descent(loss_function, init_x=init_x, lr=0.1, step_num=100)
    print(result)