import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.mnist import load_data


class Dense(tf.Module):
    def __init__(self, in_features, out_features, activation=None, name=None):
        self.activation = activation
        super().__init__(name=name)
        with self.name_scope:
            self.w = tf.Variable(tf.random.normal([in_features, out_features], stddev=0.1), name='w')
            self.b = tf.Variable(tf.zeros([out_features]), name='b')

    @tf.function
    def __call__(self, x, activation):
        return activation(tf.matmul(x, self.w) + self.b)


class SimpleDNN(tf.Module):
    def __init__(self, name):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=784, out_features=256)
        self.dense_2 = Dense(in_features=256, out_features=256)
        self.output = Dense(in_features=256, out_features=10)

    @tf.function
    def __call__(self, x):
        x = self.dense_1(x, activation=tf.nn.relu)
        x = self.dense_2(x, activation=tf.nn.relu)
        x = self.output(x, activation=tf.nn.softmax)
        return x


if __name__ == '__main__':
    (train_xs, train_ys), (test_xs, test_ys) = load_data()
    train_xs = train_xs.reshape(-1, 784)
    batch_ys = train_ys[:6]
    batch_xs = (train_xs[:6] / 255.).astype(np.float32)

    simple_dnn = SimpleDNN(name='static')
    print(simple_dnn(batch_xs))
