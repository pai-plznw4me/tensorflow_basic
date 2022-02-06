from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
import numpy as np


class Dense(Layer):
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.w, self.b = None, None

    def build(self, input_shape):
        self.w = tf.Variable(tf.random.normal([input_shape[-1], self.out_features], stddev=0.1), name='w')
        self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

    @tf.function
    def call(self, inputs, activation):
        return activation(tf.matmul(inputs, self.w) + self.b)


class DNN(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = Dense(256)
        self.dense_2 = Dense(256)

    def call(self, x):
        x = self.dense_1(x, activation=tf.nn.relu)
        return self.dense_2(x, activation=tf.nn.relu)


if __name__ == '__main__':
    (train_xs, train_ys), (test_xs, test_ys) = load_data()
    train_xs = train_xs.reshape(-1, 784)
    batch_ys = train_ys[:6]
    batch_xs = (train_xs[:6] / 255.).astype(np.float32)

    # Manipulate weights at layer level
    # Get layer weights
    dense = Dense(256, name='dynamic')
    dense(batch_xs, tf.nn.relu)
    dense_weight = dense.get_weights()[0]
    dense_bias = dense.get_weights()[1]

    # Set layer weights
    dense.set_weights([np.zeros_like(dense_weight), np.ones_like(dense_bias)])

    # Manipulate weights at Model level
    # Get weights in model
    dnn = DNN(name='dynamic')
    _ = dnn(batch_xs)  # weights 을 생성하기 위해서는 build function 을 호출해야 함
    dense_variables = dnn.get_weights()
    print(dense_weight)
    print(dense_bias)

    # Set weights zero in model
    dnn.set_weights([np.zeros_like(var) for var in dense_variables])

