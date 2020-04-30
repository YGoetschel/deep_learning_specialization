import numpy as np
from testCases import *
from gc_utils import dictionary_to_vector, vector_to_dictionary, gradients_to_vector

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def forward_propagation_n(X, Y, parameters):
    """
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    """
    m = X.shape[1]
    W1 = parameters.get('W1')
    b1 = parameters.get('b1')
    W2 = parameters.get('W2')
    b2 = parameters.get('b2')
    W3 = parameters.get('W3')
    b3 = parameters.get('b3')

    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)  # RELU ACTIVATION FX
    Z2 = np.dot(W2, A1) + b2
    A2 = np.maximum(0, Z2)  # RELU ACTIVATION FX
    Z3 = np.dot(W3, A2) + b3
    A3 = 1/(1+np.exp(-Z3))  # sigmoid activation fx

    # COMPUTE COST
    logprobs = np.multiply(Y, np.log(A3)) + np.multiply((1-Y), np.log(1-A3))
    cost = (-1/m) * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return cost, cache

def backward_propagation_n(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = (A3 - Y)
    dW3 = (1/m) * np.dot(dZ3, A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)  # returns theta, keys
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):
        theta_plus = np.copy(parameters_values)
        theta_plus[i][0] += epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))  # don't care about cache

        theta_minus = np.copy(parameters_values)                                     # Step 1
        theta_minus[i][0] -= epsilon                               # Step 2
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))

        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2. * epsilon)

        # Compare gradapprox to backward propagation gradients by computing difference.
        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
        difference = numerator / denominator

        if difference > 1e-7:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
                difference) + "\033[0m")
        else:
            print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(
                difference) + "\033[0m")

        return difference

def main():
    X, Y, parameters = gradient_check_n_test_case()
    print(type(X))
    cost, cache = forward_propagation_n(X, Y, parameters)
    gradients = backward_propagation_n(X, Y, cache)
    difference = gradient_check_n(parameters, gradients, X, Y)
if __name__=='__main__':
    main()