import tensorflow as tf
from tensorflow.python.framework import ops
from deep_learning_specialisation.Tensorflow_more_exercises.modules.create_mini_batches import random_mini_batches
import matplotlib.pyplot as plt
import numpy as np
from deep_learning_specialisation.Tensorflow_more_exercises.modules.one_hot_conversion import convert_to_one_hot
from deep_learning_specialisation.Tensorflow_more_exercises.modules.prediction import predict


def create_placeholder(n_x, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None], name='Y')
    return X, Y


def initialize_param(layers_dims):
    tf.set_random_seed(1)
    L = len(layers_dims)
    parameters = dict()

    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable(name='W' + str(l),
                                                   shape=[layers_dims[l], layers_dims[l-1]],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters['b' + str(l)] = tf.get_variable(name='b' + str(l),
                                                   shape=[layers_dims[l], 1],
                                                   initializer=tf.zeros_initializer())
    return parameters


def forward_prop(X, parameters):
    L = len(parameters) // 2
    A = X
    for l in range(1, L):
        A_prev = A
        Z = tf.add(tf.matmul(parameters.get('W' + str(l)), A_prev), parameters.get('b' + str(l)))  # Wi * Aprev + b
        A = tf.nn.relu(Z)
    # for output layer
    ZL = tf.add(tf.matmul(parameters.get('W' + str(L)), A), parameters.get('b' + str(L)))
    return ZL


def compute_loss(ZL, Y):  # apply softmax for output layer
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1500, mini_batch_size=64, print_cost=True):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)
    seed = 3
    costs = list()

    X, Y = create_placeholder(n_x=X_train.shape[0], n_y=Y_train.shape[0])

    layers_dims = [X.shape[0], 100, 100, 100, 100, 100, 10] # aiming to predict val between 0 and 9 incl
    m = X_train.shape[1]

    parameters = initialize_param(layers_dims=layers_dims)
    ZL = forward_prop(X=X, parameters=parameters)
    cost = compute_loss(ZL=ZL, Y=Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sesh:
        sesh.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = m // mini_batch_size  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1

            for mini_batch_X, mini_batch_Y in random_mini_batches(X_train, Y_train, mini_batch_size, seed=seed):
                _, minibatch_cost = sesh.run([optimizer, cost],  # don't need value from optimizer
                                             feed_dict={X: mini_batch_X,
                                                        Y: mini_batch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost==True and epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
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
        # Calculate accuracy on the train set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        return parameters


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Example of a picture
    index = 0
    plt.imshow(x_train[index], cmap='Greys')  # x is a 28px by 28 px
    plt.show()

    # Flatten the training and test images
    X_train_flatten = x_train.reshape(x_train.shape[0], -1).T
    X_test_flatten = x_test.reshape(x_test.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten / 255.
    X_test = X_test_flatten / 255.
    y_train = y_train.reshape(y_train.shape[0], -1).T
    y_test = y_test.reshape(y_test.shape[0], -1).T

    # get no of unique values in Y array
    y_set_unique_val = len(np.unique(y_train, return_counts=False))
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(y_train, y_set_unique_val )
    Y_test = convert_to_one_hot(y_test, y_set_unique_val)

    parameters = model(X_train=X_train,
                       Y_train=Y_train,
                       X_test=X_test,
                       Y_test=Y_test,
                       learning_rate=0.001,
                       num_epochs=50,
                       mini_batch_size=2**7)  # 128

    # prediction testing
    testing_idx = [0,100,2500]
    for idx in testing_idx:
        img = x_test[idx]
        plt.imshow(img, cmap='Greys')
        plt.show()
        my_image = img.reshape((1, 28 * 28)).T  # (784, 1) col vector

        my_image_prediction = predict(n_x=784, X=my_image, parameters=parameters)
        print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))


if __name__=='__main__':
    main()

