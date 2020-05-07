import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import random_mini_batches

# THIS EXAMPLE WILL FOCUS OF THE ADAM OPTIMIZER

def get_data():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral)
    plt.show()
    # reshape to size (n_x, m)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y


def create_placeholder(n_x, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None], name='Y')
    return X, Y


def initialize_parameters(layers_dims):
    L = len(layers_dims)
    parameters = dict()
    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable(name='W' + str(l),
                                                   shape=[layers_dims[l], layers_dims[l-1]],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=1)
                                                   )
        parameters['b' + str(l)] = tf.get_variable(name='b' + str(l),
                                                   shape=[layers_dims[l], 1],
                                                   initializer=tf.zeros_initializer()
                                                   )
    return parameters


def forward_prop(X, parameters):
    L = len(parameters) // 2
    A = X

    # RELU activation for hidden layer
    for l in range(1, L):
        A_prev = A
        Z = tf.add(tf.matmul(parameters.get('W' + str(l)), A_prev), parameters.get('b' + str(l)))
        A = tf.nn.relu(Z)

    # SIGMOID activation for output layer
    ZL = tf.add(tf.matmul(parameters.get('W' + str(L)), A), parameters.get('b' + str(L)))
    return ZL  # activation of output layer, proxy for Yhat


def compute_loss(ZL, Y):
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    return cost


def model(X_original, Y_original, layers_dims, learning_rate=0.0007, mini_batch_size=64, num_epochs=10000, print_cost=True):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)
    seed = 10
    costs = list()

    n_x = X_original.shape[0]
    n_y = Y_original.shape[0]
    m = X_original.shape[1]

    X, Y = create_placeholder(n_x=n_x, n_y=n_y)
    parameters = initialize_parameters(layers_dims=layers_dims)
    ZL = forward_prop(X=X, parameters=parameters)
    cost = compute_loss(ZL=ZL, Y=Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sesh:
        sesh.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / mini_batch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            mini_batches = random_mini_batches(X_original, Y_original, mini_batch_size, seed)

            for mini_batch in mini_batches:
                mini_batch_X, mini_batch_Y = mini_batch

                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, mini_batch_cost = sesh.run([optimizer, cost],  # don't need value from optimizer
                                              feed_dict={X: mini_batch_X,
                                                         Y: mini_batch_Y})
                epoch_cost += mini_batch_cost / num_minibatches

            # Print the cost every 1000 epoch
            if print_cost and epoch % 1000 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost and epoch % 100 == 0:
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
        print("Train Accuracy:", accuracy.eval({X: X_original, Y: Y_original}))
        return parameters

def main():
    train_X, train_Y = get_data()
    layers_dims = [train_X.shape[0], 5, 2, 1]

    parameters = model(X_original=train_X, Y_original=train_Y, layers_dims=layers_dims)

if __name__== '__main__':
    main()