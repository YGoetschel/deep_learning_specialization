import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

np.random.seed(1)

# load data
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Example of a picture
index = 6
plt.imshow(X_train_orig[index])
plt.show()
print("y = " + str(np.squeeze(Y_train_orig[:, index])))

# To get started, let's examine the shapes of your data.
X_train = X_train_orig/255.  # shape (m, n_h, n_w, n_c)
X_test = X_test_orig/255.   # shape (m, n_h, n_w, n_c)
Y_train = np.eye(6)[Y_train_orig.reshape(-1)]  # 6 possible classes --> shape (m, Class)
Y_test = np.eye(6)[Y_test_orig.reshape(-1)]   # shape(m, Classe)


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_H0, n_W0, n_C0], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_y])
    return X, Y


def initialize_parameters():
    kernel_dims = [[4, 4, 3, 8], [2, 2, 8, 16]]

    tf.set_random_seed(1)
    L=len(kernel_dims)
    parameters = dict()
    for l in range(L):
        parameters['W' + str(l+1)] = tf.get_variable(name='W' + str(l+1),
                                                     shape=kernel_dims[l],
                                                     initializer=tf.contrib.layers.xavier_initializer(seed=0))
    return parameters


def forward_propagation(X, parameters):
    # CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(input=X,
                      filter=parameters.get('W1'),
                      strides=[1,1,1,1],  # [1,s,s,1]
                      padding="SAME")
    # NOW APPLY ACTIVATION FX
    A1 = tf.nn.relu(features=Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(value=A1,
                        ksize=[1,8,8,1],  # [1,f,f,1]
                        strides=[1,8,8,1], # [1,s,s,1]
                        padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(input=P1,
                      filter=parameters.get('W2'),
                      strides=[1,1,1,1],
                      padding='SAME')
    # APPLY ACTIVATION FX = relu
    A2 = tf.nn.relu(features=Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(value=A2,
                        ksize=[1,4,4,1],
                        strides=[1,4,4,1],
                        padding='SAME')
    # FLATTEN
    P2_flattened = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED
    Z3 = tf.contrib.layers.fully_connected(P2_flattened,
                                           6,  # 6 possible output classes
                                           activation_fn=None)
    return Z3


def compute_cost(ZL, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ZL,labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009, num_epochs=100, minibatch_size=64, print_cost=True):
    ops.reset_default_graph()               # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                   # to keep results consistent (tensorflow seed)
    seed = 3                                # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0=64, n_W0=64, n_C0=3, n_y=6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X=X, parameters=parameters)
    cost = compute_cost(ZL=Z3, Y=Y)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = m // minibatch_size  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1

            for mini_batch_X, mini_batch_Y in random_mini_batches(X_train, Y_train, minibatch_size, seed=seed):
                _, minibatch_cost = sess.run([optimizer, cost], {X: mini_batch_X, Y: mini_batch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


def main():
    _, _, parameters = model(X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()

