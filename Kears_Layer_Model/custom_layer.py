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

    def get_config(self):
        return {"out_features": self.out_features}


if __name__ == '__main__':
    (train_xs, train_ys), (test_xs, test_ys) = load_data()
    train_xs = train_xs.reshape(-1, 784)
    batch_ys = train_ys[:6]
    batch_xs = (train_xs[:6] / 255.).astype(np.float32)

    dense = Dense(256)
    config = dense.get_config()

    Dense.from_config(config)
