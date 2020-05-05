import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

"""
PARAMETER DEFINITION
"""
def create_cost_fonction():
    w = tf.Variable(0, dtype=tf.float32)
    x = tf.placeholder(tf.float32, [3, 1])  # x is 3row, 1col array of data whose values are assigned later
    """Tensorflow figures out the back prop on its own, so only need to define de forward propagation"""
    cost_fx = x[0][0]*w**2 + x[1][0]*w + x[2][0]
    return cost_fx, w, x


def minimize_loss(cost_fx):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01,
                                                  use_locking=False,
                                                  name='GradientDescent')
    train = optimizer.minimize(cost_fx)
    return train


def run_single_iteration_gradient_descent(session, train, x, w, coefficient):
    session.run(train, feed_dict={x: coefficient})  # get data into the cost fx
    print(session.run(w))


def run_many_iter_gradient_descent(session, train, x, w, coefficient):
    for i in range(1000):
        session.run(train, feed_dict={x: coefficient}) # get data into the cost fx
    print(session.run(w))

def main():
    coefficient = np.array([[1.], [-10.], [25.]])  # col verctor (3,1)
    cost_fx, w, x = create_cost_fonction()
    train = minimize_loss(cost_fx)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    run_single_iteration_gradient_descent(session=session,
                                          train=train,
                                          x=x,
                                          w=w,
                                          coefficient=coefficient)

    run_many_iter_gradient_descent(session=session,
                                   train=train,
                                   x=x,
                                   w=w,
                                   coefficient=coefficient)

if  __name__ == '__main__':
    main()

