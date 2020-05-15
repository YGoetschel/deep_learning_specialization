import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from cnn_utils import random_mini_batches


def prepare_data(X_train, y_train, X_test, y_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize input values
    X_train = X_train / 255
    X_test = X_test / 255

    # Reshape the dataset into 4D array
    X_train = X_train.reshape(X_train.shape[0], 28,28,1)  # -> (m,n_h, n_w, n_c)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # one hot encoding output values
    y_train = np.eye(10)[y_train.reshape(-1)]
    y_test = np.eye(10)[y_test.reshape(-1)]
    return X_train, y_train, X_test, y_test


def create_placeholder(n_h, n_w, n_c, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_h, n_w, n_c], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_y], name='Y')
    return X, Y


def initialize_parameters():
    tf.set_random_seed(1)
    kernels_dims = [[5, 5, 1, 6], [5, 5, 6, 16]]  # kernel height, kernel width, channels(l-1),channels(l)
    L = len(kernels_dims)
    parameters = dict()

    for l in range(L):
        parameters['W' + str(l+1)] = tf.get_variable(name='W' + str(l+1),
                                                     shape=kernels_dims[l],
                                                     initializer=tf.contrib.layers.xavier_initializer(seed=0))
    return parameters


def forward_prop(X, parameters):
    """
    Model architecture:
    CONV2D -> tanh -> AVG POOL -> CONV2D -> tanh -> AVG POOL -> FLATTEN -> FC-> FC-> SOFTMAX
    Padding is added to first conv layer in order to match the output shape of original LeNet-5 architecture (28*28*6)
    mnist img shape is (28*28*1) hence padding added to maintain n_h = 28

    """
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(input=X,
                      filter=parameters.get('W1'),
                      strides=[1, 1, 1, 1],  # [1,s,s,1]
                      padding='SAME')
    # TANH Activation fx
    A1 = tf.nn.tanh(Z1)
    # AVGPOOL: window 2x2, stride 2, padding 'valid'
    P1 = tf.nn.avg_pool(value=A1,
                        ksize=[1, 2, 2, 1],  # [1,f,f,1]
                        strides=[1, 2, 2, 1],  # [1,s,s,1]
                        padding='VALID')
    # CONV2D: stride of 1, padding 'valid'
    Z2 = tf.nn.conv2d(input=P1,
                      filter=parameters.get('W2'),
                      strides=[1, 1, 1, 1],
                      padding='VALID')
    # TANH Activation fx
    A2 = tf.nn.tanh(Z2)
    # AVGPOOL: window 2x2, stride 2, padding 'valid'
    P2 = tf.nn.avg_pool(value=A2,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='VALID')
    # FLATTEN
    P2_flattened = tf.contrib.layers.flatten(P2)
    # Fully connected layer #nodes = 120
    FC3 = tf.contrib.layers.fully_connected(P2_flattened, 120, activation_fn=tf.nn.tanh)
    # Fully connected layer #nodes = 84
    FC4 = tf.contrib.layers.fully_connected(FC3, 84, activation_fn=tf.nn.tanh)
    # output layer
    FC5 = tf.contrib.layers.fully_connected(FC4, 10, activation_fn=None)
    return FC5


def compute_loss(ZL, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=ZL))
    return cost


def model(x_train, y_train, x_test, y_test, learning_rate=0.009, num_epochs=10, minibatch_size=128, print_cost=True):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)
    seed = 3
    costs = list()

    X_train, y_train, X_test, y_test = prepare_data(x_train, y_train, x_test, y_test)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = y_train.shape[1]

    X, Y = create_placeholder(n_h=n_H0, n_w=n_W0, n_c=n_C0, n_y=n_y)
    parameters = initialize_parameters()
    ZL = forward_prop(X=X, parameters=parameters)
    cost = compute_loss(ZL, Y=Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = m // minibatch_size  # nb of full minibatches
            seed = seed + 1
            for mini_batch_X, mini_batch_Y in random_mini_batches(X_train, y_train, minibatch_size, seed):
                _, minibatch_cost = sess.run([optimizer, cost], {X: mini_batch_X, Y: mini_batch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost==True and epoch % 2 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost==True and epoch % 1 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(ZL, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Example of a picture
    index = 0
    plt.imshow(x_train[index], cmap='Greys')  # x is a 28px by 28 px
    plt.show()

    _, _, parameters = model(x_train, y_train, x_test, y_test)
if __name__ == "__main__":
    main()
