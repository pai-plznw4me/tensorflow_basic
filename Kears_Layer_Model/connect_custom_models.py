from tensorflow.keras.layers import Layer, Input
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


class DNN1(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = Dense(256)
        self.dense_2 = Dense(256)

    def call(self, x):
        x = self.dense_1(x, activation=tf.nn.relu)
        return self.dense_2(x, activation=tf.nn.relu)


class DNN2(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = Dense(256)
        self.dense_2 = Dense(10)

    def call(self, x):
        x = self.dense_1(x, activation=tf.nn.relu)
        return self.dense_2(x, activation=tf.nn.relu)


if __name__ == '__main__':
    (train_xs, train_ys), (test_xs, test_ys) = load_data()
    train_xs = train_xs.reshape(-1, 784)
    batch_ys = train_ys[:6]
    batch_xs = (train_xs[:6] / 255.).astype(np.float32)

    dense = Dense(256, name='dynamic')

    # DNN Model
    input_ = Input(shape=(784,))
    dnn1 = DNN1('dynamic_dnn_1')
    dnn2 = DNN2('dynamic_dnn_2')

    z1 = dnn1(batch_xs)
    z2 = dnn2(z1)
    print(z2)


