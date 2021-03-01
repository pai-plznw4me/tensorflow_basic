import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
# tensorflow 2.x Graph Execution

# load MNIST Dataset
train_data, test_data = tf.keras.datasets.mnist.load_data()
(train_xs, train_ys) = train_data
train_xs = train_xs.reshape([-1, 784])
train_xs = train_xs.astype(np.float32)

# Generate W, b
def generate_w_b(in_, out):
    w_init = tf.random.normal([in_, out])
    w = tf.Variable(w_init, dtype=tf.float32)
    b_init = tf.zeros(out)
    b = tf.Variable(b_init,dtype=tf.float32)
    return w, b

w1, b1 = generate_w_b(784, 128)
w2, b2 = generate_w_b(128, 128)
w3, b3 = generate_w_b(128, 128)
w4, b4 = generate_w_b(128, 10)

# Run Dense Layer, 100 times
@tf.function
def graph_execution(layer):
    z1 = tf.matmul(layer, w1) + b1
    a1 = tf.nn.relu(z1)

    z2 = tf.matmul(a1, w2) + b2
    a2 = tf.nn.relu(z2)

    z3 = tf.matmul(a2, w3) + b3
    a3 = tf.nn.relu(z3)

    logits = tf.matmul(a3, w4) + b4

# execute Graph, 100 times
s = time.time()
for i in tqdm(range(100)):
    graph_execution(train_xs)
print('Consume time : {}'.format(time.time() - s))