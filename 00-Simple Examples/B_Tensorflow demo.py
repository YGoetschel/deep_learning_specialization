import numpy as np
import tensorflow as tf

"""
PARAMETER DEFINITION
"""
w = tf.Variable(0, dtype = tf.float32)
cost_fx = tf.add(tf.add(w**2, tf.multiply(-10, w)), 25) # tf.add () only adds 2 elements together, if 3 elements then use tf.add twice

optimizer = tf.optimizers.SGD(learning_rate=0.001)
# optimizer = tf.train.GradientDescentOptimizer(0.01)  # learning algo to min cost_fx
train = optimizer.minimize(cost_fx)

init = tf.global_variable_initializer()
session = tf.Session()
session.run(init)
session.run(w)

# perform 1 step of gradient descent and then print value
session.run(train)
print(session.run(w))