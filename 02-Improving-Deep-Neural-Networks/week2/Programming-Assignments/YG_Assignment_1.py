import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import math

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# ---------------------------------------
# LOAD DATA
# ---------------------------------------
def load_data():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral)
    plt.show()
    # reshape to size (n_x, m)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y

# ---------------------------------------------------
# Initialize parameters and optimizer
# ---------------------------------------------------
def initialize_parameters(layer_dims):
    parameters = dict()
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def initialize_optimizer(parameters, optimizer):
    L = len(parameters) // 2  # nb of layers in NN
    v = dict()
    s = dict()

    # optimizer for each layer in the neural network
    if optimizer == 'momentum':
        for l in range(L):
            v['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape) # Vdw matrice has same size as W
            v['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
    elif optimizer == 'adam':
        for l in range(L):
            v['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
            v['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
            s['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
            s['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
    return v, s

# ---------------------------------------------------
# create mini batches
# ---------------------------------------------------
def random_mini_batches(X, y, size, seed=0):
    np.random.seed(seed)  # to get same results as solutions
    m = X.shape[1]
    mini_batches = list()

    # shuffle X, y
    permutations = list(np.random.permutation(m))
    shuffled_X = X[:, permutations]
    shuffled_y = y[:, permutations].reshape(1, m)  # make sure vector is of dim (1, m)

    num_of_complete_minibatches = math.floor((m / size))  # might be some left over, --> imcomplete set
    for k in range(0, num_of_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * size : (k+1) * size]  # --> rows : cols
        mini_batch_y = shuffled_y[:, k * size : (k+1) * size]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    # handling left over
    if m % size != 0:
        mini_batch_X = shuffled_X[:, num_of_complete_minibatches * size : ]
        mini_batch_y = shuffled_y[:, num_of_complete_minibatches * size : ]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches

# ---------------------------------------------------
# Implement forward propagation
# ---------------------------------------------------
def linear_prop(W, b, A_prev, activation):
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)
    activation_cache = Z

    if activation == 'relu':
        A = np.maximum(0, Z)
    elif activation == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    caches = (linear_cache, activation_cache)
    return A, caches

def forward_prop(X, parameters):
    L = len(parameters) // 2
    A = X
    f_prop_cache = list()

    for l in range(1, L):
        A_prev = A
        A, caches = linear_prop(W=parameters['W' + str(l)],
                                b=parameters['b' + str(l)],
                                A_prev=A_prev,
                                activation='relu')
        f_prop_cache.append(caches)

    AL, caches = linear_prop(W=parameters['W' + str(L)],
                             b=parameters['b' + str(L)],
                             A_prev=A,  # A value returned on last iteration of hidden layer
                             activation='sigmoid')
    f_prop_cache.append(caches)

    return AL, f_prop_cache


# ---------------------------------------------------
# compute loss fx --> no regularization
# ---------------------------------------------------
def compute_loss(AL, Y):
    m = Y.shape[1]
    logprobs = np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL))
    cost = (-1/m) * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

# ---------------------------------------------------
# Implement backward propagation
# ---------------------------------------------------
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    Z = activation_cache

    def linear_backward(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m) * (np.dot(dZ, A_prev.T))
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    if activation == 'relu':
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
    elif activation == 'sigmoid':
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)

    return linear_backward(dZ=dZ, cache=linear_cache)

def backward_prop(AL, Y, caches):
    grads = dict()
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # init aAL --> derivative of loss fx
    dAL = -(np.divide(Y, AL) - np.divide((1-Y), (1-AL)))
    # output output layer
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dA=dAL,
                                                                                                  cache=caches[-1],
                                                                                                  activation='sigmoid')
    # for hidden layer
    for l in reversed(range(L - 1)):
        grads['dA' + str(l+1)], grads['dW' + str(l+1)], grads['db' + str(l+1)] = linear_activation_backward(dA=grads['dA'+str(l+2)],
                                                                                                            cache=caches[l],
                                                                                                            activation='relu')
    return grads


# ---------------------------------------------------
# update paramters --> momentim or adam optizimer based
# ---------------------------------------------------
def update_param_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        v['dW' + str(l + 1)] = (beta * v['dW' + str(l + 1)]) + ((1-beta) * grads['dW' + str(l + 1)])
        v['db' + str(l + 1)] = (beta * v['db' + str(l + 1)]) + ((1 - beta) * grads['db' + str(l + 1)])

        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * v['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v['db' + str(l + 1)]
    return parameters, v

def update_param_with_adam(parameters, grads, v, s, t,  beta1, beta2, epsilon, learning_rate):
    L = len(parameters) // 2

    v_corrected = dict()
    s_corrected = dict()

    for l in range(L):
        v['dW' + str(l + 1)] = (beta1 * v['dW' + str(l + 1)]) + ((1-beta1) * grads['dW' + str(l + 1)])
        v['db' + str(l + 1)] = (beta1 * v['db' + str(l + 1)]) + ((1 - beta1) * grads['db' + str(l + 1)])
        v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - np.power(beta1, t))

        s['dW' + str(l + 1)] = (beta2 * s['dW' + str(l + 1)]) + ((1-beta2) * np.power(grads['dW' + str(l + 1)], 2))
        s['db' + str(l + 1)] = (beta2 * s['db' + str(l + 1)]) + ((1-beta2) * np.power(grads['db' + str(l + 1)], 2))
        s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - np.power(beta2, t))

        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * v_corrected['dW' + str(l + 1)] / np.sqrt(s_corrected['dW' + str(l + 1)] + epsilon)
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v_corrected['db' + str(l + 1)] / np.sqrt(s_corrected['db' + str(l + 1)] + epsilon)
    return parameters, v, s


# ----------------------------------------------------------
# Aggregate all in model function
# ----------------------------------------------------------
def model(X, Y, layer_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta1=0.9, beta2=0.999,epsilon=1e-8,
          num_epochs=10000, print_cost=True):
    seed = 10
    t = 0
    costs = list()

    # start model
    parameters = initialize_parameters(layer_dims=layer_dims)
    v, s = initialize_optimizer(parameters=parameters, optimizer=optimizer)

    # optimization loop
    for i in range(num_epochs):  # nb of passes over whole dataset
        seed = seed + 1
        mini_batches = random_mini_batches(X=X, y=Y, size=mini_batch_size, seed=seed)

        for mini_batch in mini_batches:  # each mini batch has X, y
            mini_batch_X, mini_batch_Y = mini_batch

            AL, f_prop_cache = forward_prop(X=mini_batch_X, parameters=parameters)
            cost = compute_loss(AL=AL, Y=mini_batch_Y)
            grads = backward_prop(AL=AL, Y=mini_batch_Y, caches=f_prop_cache)

            # update_parameters
            if optimizer == 'momentum':
                parameters, v = update_param_with_momentum(parameters=parameters,
                                                           grads=grads,
                                                           v=v,
                                                           beta=beta1,
                                                           learning_rate=learning_rate)
            elif optimizer == 'adam':
                t = t + 1
                parameters, v, s = update_param_with_adam(parameters=parameters,
                                                          grads=grads,
                                                          v=v,
                                                          s=s,
                                                          t=t,
                                                          beta1=beta1,
                                                          beta2=beta2,
                                                          epsilon=epsilon,
                                                          learning_rate=learning_rate)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

def main():
    train_X, train_Y = load_data()
    layer_dims = [train_X.shape[0], 5, 2, 1]
    # optimizer = 'momentum'
    optimizer = 'adam'

    parameters = model(X=train_X, Y=train_Y, layer_dims=layer_dims, optimizer=optimizer)

if __name__ == '__main__':
    main()

