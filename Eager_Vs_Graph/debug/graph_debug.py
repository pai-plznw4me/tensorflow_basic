import tensorflow as tf

# run tensorflow 1.x
assert float(tf.__version__[0]) < 2, print('tensorflow version' , tf.__version__)

a = tf.constant(3)
b = tf.constant(5)
c = a + b
