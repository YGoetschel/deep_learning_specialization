import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from tf_utils import *
import matplotlib.pyplot as plt


def get_data():
    # Loading the dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # Example of a picture
    index = 0
    plt.imshow(X_train_orig[index])
    plt.show()
    print("y = " + str(np.squeeze(Y_train_orig[:, index])))
    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    return X_train, Y_train, X_test, Y_test


def create_placeholders(n_x, n_y):
    """
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    """
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')
    return X, Y


def initialize_parameters(layers_dims):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    seed = tf.set_random_seed(1)
    parameters = dict()
    # xavier_initializer = np.random.randn(layers_dim[l], layers_dim[l-1]) * np.sqrt(1/layers_dim[l-1])
    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = tf.get_variable("W"+str(l),
                                                   [layers_dims[l], layers_dims[l-1]],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters["b" + str(l)] = tf.get_variable("b"+str(l),
                                                   [layers_dims[l], 1],
                                                   initializer=tf.zeros_initializer())
    return parameters


def forward_propagation(X, parameters):
    L = len(parameters) // 2
    A = X
    for l in range(1, L):
        A_prev = A
        Z = tf.add(tf.matmul(parameters.get("W" + str(l)), A_prev), parameters.get("b" + str(l)))  # W1*X+b1 for layer 1
        A = tf.nn.relu(Z)
    # final layer
    ZL = tf.add(tf.matmul(parameters.get("W" + str(L)), A), parameters.get("b" + str(L)))
    return ZL


def compute_cost(ZL, Y):
    """
    apply softmax fx
    --> the `logits` and `labels` inputs of `tf.nn.softmax_cross_entropy_with_logits`
    are expected to be of shape (number of examples, num_classes).
    We have thus transposed ZL and Y.
    """
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=labels))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)
    seed = 3
    costs = list()

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x=X_train.shape[0], n_y=Y_train.shape[0])

    layers_dims = [X.shape[0], 25, 12, 6]
    m = X_train.shape[1]

    # Initialize parameters
    parameters = initialize_parameters(layers_dims=layers_dims)
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ZL = forward_propagation(X=X, parameters=parameters)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(ZL=ZL, Y=Y)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sesh:
        sesh.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            mini_batches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sesh.run([optimizer, cost],  # don't need value from optimizer
                                             feed_dict={X: mini_batch_X,
                                                        Y: mini_batch_Y})

                epoch_cost += minibatch_cost / num_minibatches

                # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

            # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sesh.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        return parameters


def main():
    X_train, Y_train, X_test, Y_test = get_data()
    parameters = model(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
if __name__ == '__main__':
    main()