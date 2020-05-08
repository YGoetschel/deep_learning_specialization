import tensorflow as tf


def predict(n_x, X, parameters):
    L = len(parameters) // 2
    params = dict()
    for l in range(1, L+1):  # as need last layer to be included
        params['W' + str(l)] = tf.convert_to_tensor(parameters['W' + str(l)])
        params['b' + str(l)] = tf.convert_to_tensor(parameters['b' + str(l)])

    x = tf.placeholder("float", [n_x, 1])

    ZL = forward_propagation_for_predict(X=x, parameters=params)
    p = tf.argmax(ZL)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})
    return prediction


def forward_propagation_for_predict(X, parameters):
    L = len(parameters) // 2
    A = X
    for l in range(1, L):
        A_prev = A
        Z = tf.add(tf.matmul(parameters.get('W' + str(l)), A_prev), parameters.get('b' + str(l)))  # Wi * Aprev + b
        A = tf.nn.relu(Z)
    # for output layer
    ZL = tf.add(tf.matmul(parameters.get('W' + str(L)), A), parameters.get('b' + str(L)))
    return ZL