import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.datasets.mnist import load_data


class Dense(Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.in_features = in_features

        self.w = tf.Variable(tf.random.normal([self.in_features, self.out_features], stddev=0.1), name='w')
        self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

    @tf.function
    def __call__(self, x, activation):
        return activation(tf.matmul(x, self.w) + self.b)


class DNN(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)

        self.dense_1 = Dense(in_features=784, out_features=256)
        self.dense_2 = Dense(in_features=256, out_features=2)

    def call(self, x):
        x = self.dense_1(x, activation=tf.nn.relu)
        return self.dense_2(x, activation=tf.nn.relu)

    def get_config(self):
        return {'name': self.name}


if __name__ == '__main__':
    (train_xs, train_ys), (test_xs, test_ys) = load_data()
    train_xs = train_xs.reshape(-1, 784)
    batch_ys = train_ys[:6]
    batch_xs = (train_xs[:6] / 255.).astype(np.float32)

    # Dense Layer
    dense = Dense(784, 256, name='static')
    print(dense(batch_xs, activation=tf.nn.relu))

    # DNN Model
    dnn = DNN('static dnn')
    print(dnn(batch_xs))
