import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.mnist import load_data


class Dense(tf.Module):
    def __init__(self, out_features, name=None):
        super().__init__(name=name)
        self.is_built = False
        self.out_features = out_features

    @tf.function
    def __call__(self, x, activation):
        if not self.is_built:
            self.w = tf.Variable(
                tf.random.normal([x.shape[-1], self.out_features], stddev=0.1), name='w')
            self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
            self.is_built = True

        return activation(tf.matmul(x, self.w) + self.b)


class SimpleDNN(tf.Module):
    def __init__(self, name):
        super().__init__(name=name)

        self.dense_1 = Dense(out_features=256)
        self.dense_2 = Dense(out_features=256)
        self.output = Dense(out_features=10)

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

    simple_dnn = SimpleDNN(name='dynamic')
    print(simple_dnn(batch_xs))
