import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

"""
2 - Outline of the Assignment
To build your neural network, you will be implementing several "helper functions". 

- Initialize the parameters for a two-layer network and for an L-layer neural network.
- Implement the forward propagation module (shown in purple in the figure below).
     - Complete the LINEAR part of a layer's forward propagation step (resulting in $Z^{[l]}$).
     - We give you the ACTIVATION function (relu/sigmoid).
     - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
     - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). This gives you a new L_model_forward function.
- Compute the loss.
- Implement the backward propagation module (denoted in red in the figure below).
    - Complete the LINEAR part of a layer's backward propagation step.
    - We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward) 
    - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
    - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
- Finally update the parameters.
"""

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = dict()
    L = len(layer_dims)
    
    for l in range(1, L):  # idx=0 refers to input variables x which does not have W and b.
        parameters['W{}'.format(str(l))] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b{}'.format(str(l))] = np.zeros((layer_dims[l], 1))
    return parameters


# ## Forward Propagation
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


# implement relu activation fx in hidden layer and sigmoid in output layer
def linear_activation_forward(A_prev, W, b, activation):
    # assume using RELU fx in hidden layer and SIGMOID in output layer
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A=A_prev, W=W, b=b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A=A_prev, W=W, b=b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # //= floor division --> parameters contain /W/ and /b/ for each layer
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev=A_prev,
                                             W=parameters.get('W{}'.format(str(l))),
                                             b=parameters.get('b{}'.format(str(l))),
                                             activation='relu')
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A_prev = A,
                                          W = parameters.get('W{}'.format(str(L))),
                                          b = parameters.get('b{}'.format(str(L))),
                                          activation = 'sigmoid')
    
    caches.append(cache)
    # vector 𝐴𝐿 contains your predictions
    return AL, caches


# ## Compute Cost function
def compute_cost(AL, Y):
    m = Y.shape[1]
    logprobs = np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL))
    cost = (-1/m) * np.sum(logprobs)
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation --> DERIVATIVE OF THE COST FUNCTION
    dAL = - (np.divide(Y, AL) - np.divide((1-Y),(1-AL)))
    
    current_cache = caches[-1]  # last storred cache
    
    grads["dA{}".format(str(L))], grads["dW{}".format(str(L))], grads["db{}".format(str(L))] = linear_activation_backward(dAL, 
                                                                                                                          current_cache, 
                                                                                                                          activation="sigmoid"
                                                                                                                         )
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        W = "W{}".format(str(l+1))
        b = "b{}".format(str(l+1))
        dW = "dW{}".format(str(l+1))
        db = "db{}".format(str(l+1))
        
        parameters[W] = parameters.get(W) - learning_rate * grads[dW]
        parameters[b] = parameters.get(b) - learning_rate * grads[db]
    return parameters

