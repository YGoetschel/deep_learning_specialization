import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import skimage
from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
print(train_x_orig.shape)
print(test_y.shape)
# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
plt.show()
print("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# reshape train & test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T  # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T
# standardize
train_x = train_x_flatten/255
test_x = test_x_flatten/255
print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

# ## L-layer Neural Network
layer_dims = [12288, 20, 7, 5, 1] #  5-layer model


def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = dict()
    L = len(layer_dims)
    
    for l in range(1, L):
        W = 'W{}'.format(str(l))
        b = 'b{}'.format(str(l))
        
        parameters[W] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])# * 0.01
        parameters[b] = np.zeros((layer_dims[l], 1))
        
    return parameters


# ### Implement forward prop
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    assert(Z.shape == (W.shape[0], A.shape[1]))
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape==(W.shape[0], A_prev.shape[1]))
    caches = (linear_cache, activation_cache)
    return A, caches


def L_model_forward(X, parameters):
    """ 
    A0=X which is the input units
    AL = last post activation value
    """
    caches = list()
    A = X  # 
    L = len(parameters) // 2 # as 2 params per layer (W,b)

    # for hidden layer
    for l in range(1, L):        
        A_prev = A
        W = parameters.get('W{}'.format(str(l)))
        b = parameters.get('b{}'.format(str(l)))
        
        A, cache = linear_activation_forward(A_prev,
                                             W,
                                             b,
                                             activation='relu')
        caches.append(cache)

    # for output layer
    AL, cache = linear_activation_forward(A, 
                                          parameters.get('W{}'.format(str(L))),
                                          parameters.get('b{}'.format(str(L))),
                                          activation='sigmoid')
    caches.append(cache)
    return AL, caches


# ### Compute cost function
def compute_cost(AL, Y):
    m = Y.shape[1]
    logprobs = np.multiply(Y, np.log(AL)) + np.multiply((1-Y),np.log(1-AL))
    cost = -1/m * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost


# ### Implementing backward prop
def linear_backward(dZ, cache):
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
    
    # initialze AL by taking derivative of loss function
    dAL = -(np.divide(Y, AL)-np.divide((1-Y), (1-AL)))

    # starting at AL(hence, Yhat)
    grads["dA{}".format(str(L))], grads["dW{}".format(str(L))], grads["db{}".format(str(L))] = linear_activation_backward(dAL,
                                                                                                                          caches[-1],
                                                                                                                          'sigmoid')
    # work backwards                                                                                                                      
    for l in reversed(range(L-1)):
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 2)], caches[l], 'relu')
        grads["dA{}".format(str(l+1))] = dA_prev
        grads["dW{}".format(str(l+1))] = dW
        grads["db{}".format(str(l+1))] = db
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


# ## RUNNING THE MODEL

def L_layer_model(X, Y, layer_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    costs = list()
    parameters = initialize_parameters_deep(layer_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)   # perform forward prop
        cost = compute_cost(AL, Y)                    # compute the cost
        grads = L_model_backward(AL, Y, caches)       # perform backward prop
        parameters = update_parameters(parameters,    # update parameters
                                       grads, 
                                       learning_rate=learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


# ### Run it!!
parameters = L_layer_model(train_x, 
                           train_y, 
                           layer_dims, 
                           num_iterations = 2500, 
                           print_cost = True)

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
