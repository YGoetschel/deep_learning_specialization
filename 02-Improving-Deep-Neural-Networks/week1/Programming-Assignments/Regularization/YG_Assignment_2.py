# The AIM is to implement regularization to AVOID OVERFITTING of the dataset

import numpy as np
import matplotlib.pyplot as plt
from reg_utils import load_2D_dataset
plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# ----------------------------------------------------
# initialize params
# ----------------------------------------------------
def initialize_params(layers_dims):
    np.random.seed(3)
    L = len(layers_dims)
    parameters = dict()
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

# ----------------------------------------------------
# implement forward propagation
# ----------------------------------------------------
def linear_forward_activation(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)
    activation_cache = Z

    if activation == 'relu':
        A = np.maximum(0, Z)
    elif activation == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    caches = (linear_cache, activation_cache)
    return A, caches

def model_forward_prop(X, parameters):
    A = X
    L = len(parameters) // 2
    f_prop_cache = list()

    # loop over hidden layer
    for l in range(1, L):
        A_prev = A
        A, cache = linear_forward_activation(A_prev=A_prev,
                                             W=parameters.get('W' + str(l)),
                                             b=parameters.get('b' + str(l)),
                                             activation='relu')
        f_prop_cache.append(cache)

    # compute AL --> Yhat
    AL, cache = linear_forward_activation(A_prev=A,
                                          W=parameters.get('W' + str(L)),
                                          b=parameters.get('b' + str(L)),
                                          activation='sigmoid')
    f_prop_cache.append(cache)
    return AL, f_prop_cache

# ----------------------------------------------------
# compute loss function applying L2 regularization
# ----------------------------------------------------
def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters.get('W1')
    W2 = parameters.get('W2')
    W3 = parameters.get('W3')

    # compute std cross_entropy_cost
    logprobs = np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))
    cross_entropy_cost = (-1 / m) * np.sum(logprobs)
    L2_regularization_cost = (1 / m) * (lambd / 2) * (np.sum(np.square(W1)) +
                                                      np.sum(np.square(W2)) +
                                                      np.sum(np.square(W3))
                                                      )
    cost = cross_entropy_cost + L2_regularization_cost
    return np.squeeze(cost)

# ----------------------------------------------------
# Implement backward propagation applying L2 regularization
# ----------------------------------------------------
def linear_activation_backward(dA, cache, lambd, activation):
    linear_cache, activation_cache = cache
    Z = activation_cache

    def linear_backward(dZ, cache, lambd):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1 / m) * (np.dot(dZ, A_prev.T) + lambd * W)  # adding regularization term to gradient descent
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    if activation == 'relu':
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
    elif activation == 'sigmoid':
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
    return linear_backward(dZ=dZ, cache=linear_cache, lambd=lambd)

def backward_propagation_with_regularization(AL, Y, cache, lambd):
    m = Y.shape[1]
    L = len(cache)
    # Y = Y.reshape(AL.shape)
    grads = dict()

    # init dAL
    dAL = -(np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))

    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dA=dAL,
                                                                                                  cache=cache[-1],
                                                                                                  lambd=lambd,
                                                                                                  activation='sigmoid')
    for l in reversed(range(L - 1)):
        grads['dA' + str(l + 1)], grads['dW' + str(l + 1)], grads['db' + str(l + 1)] = linear_activation_backward(
            dA=grads['dA' + str(l + 2)],
            cache=cache[l],
            lambd=lambd,
            activation='relu')
    return grads

# ----------------------------------------------------
# update parameters
# ----------------------------------------------------
def update_parameters(grads, parameters, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads.get('dW' + str(l + 1))
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads.get('db' + str(l + 1))
    return parameters

# ----------------------------------------------------
# aggregate all under function called: Model
# ----------------------------------------------------
def model(X, Y, learning_rate=0.3, num_iterations=15000, print_cost=True, lambd=0):
    layers_dims = [X.shape[0], 20, 3, 1]
    costs_all = list()

    parameters = initialize_params(layers_dims=layers_dims)
    for i in range(0, num_iterations):
        AL, f_prop_cache = model_forward_prop(X=X, parameters=parameters)
        cost = compute_cost_with_regularization(AL=AL, Y=Y, parameters=parameters, lambd=lambd)
        grads = backward_propagation_with_regularization(AL=AL, Y=Y, cache=f_prop_cache, lambd=lambd)
        parameters = update_parameters(grads=grads, parameters=parameters, learning_rate=learning_rate)

        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(str(i), str(cost)))
        if print_cost and i % 1000 == 0:
            costs_all.append(cost)

    # plot the cost
    plt.plot(costs_all)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

def main():
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    parameters = model(train_X, train_Y, num_iterations=30000, lambd=0.7)
if __name__ == '__main__':
    main()