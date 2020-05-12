import tensorflow as tf
import numpy as np
import warnings
# tf.disable_v2_behavior()
warnings.filterwarnings(action="ignore", category=FutureWarning)
np.random.seed(1)

def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """
    np.random.seed(1)
    X = tf.constant(np.random.randn(3, 1), name='X')
    W = tf.constant(np.random.randn(4, 3), name='W')
    b = tf.constant(np.random.randn(4, 1), name='b')
    Y = tf.add(tf.matmul(W, X), b)  # perform matrix multiplication

    with tf.Session() as session:
        result = session.run(Y)

    return result


def sigmoid(z):
    x = tf.placeholder(tf.float32, name='x')
    sigmoid = tf.sigmoid(x)
    with tf.Session() as session:
        result = session.run(sigmoid, feed_dict={x:z})
    return result


def compute_loss(logits, labels):
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,  labels=y)
    with tf.Session() as session:
        cost = session.run(cost, feed_dict={z:logits,
                                            y:labels})
    return cost


def one_hot_matrix(labels, C):
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    with tf.Session() as session:
        one_hot = session.run(one_hot_matrix)
    return one_hot


def main():

    label_y = np.array([0, 0, 1, 1])
    Z = linear_function()
    A = sigmoid(z=Z)
    cost = compute_loss(logits=A, labels=label_y)
    print(cost)

    labels = np.array([1, 2, 3, 0, 2, 1])
    one_hot = one_hot_matrix(labels, C=4)
    print("one_hot = " + str(one_hot))
if __name__ == '__main__':
    main()