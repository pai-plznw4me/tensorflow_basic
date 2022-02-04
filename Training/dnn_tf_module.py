import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


class Dense(tf.Module):
    def __init__(self, in_features, out_features, activation=None, name=None):
        self.activation = activation
        super().__init__(name=name)
        with self.name_scope:
            self.w = tf.Variable(tf.random.normal([in_features, out_features], stddev=0.1), name='w')
            self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        if self.activation == 'softmax':
            return tf.nn.softmax(y)
        elif self.activation == 'relu':
            return tf.nn.relu(y)
        elif self.activation is None:
            return y
        else:
            raise NotImplementedError


# class FlexibleDenseModule(tf.Module):
#     def __init__(self, out_features, name=None):
#         super().__init__(name=name)
#         self.is_built = False
#         self.out_features = out_features
#
#     def __call__(self, x):
#         if not self.is_built:
#             self.w = tf.Variable(
#                 tf.random.normal([x.shape[-1], self.out_features]), name='w')
#             self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
#             self.is_built = True
#
#         y = tf.matmul(x, self.w) + self.b
#         if self.activation == 'softmax':
#             return tf.nn.softmax(y)
#         elif self.activation == 'relu':
#             return tf.nn.relu(y)
#         elif self.activation is None:
#             return y
#         else:
#             raise NotImplementedError


class SimpleDNN(tf.Module):
    def __init__(self, name):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=784, out_features=256, activation='relu')
        self.dense_2 = Dense(in_features=256, out_features=256, activation='relu')
        self.output = Dense(in_features=256, out_features=10, activation='softmax')

    def __call__(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.output(x)
        return x


def crossentropy(y, y_hat):
    y = tf.clip_by_value(y, 1e-15, 0.99)
    y_hat = tf.clip_by_value(y_hat, 1e-15, 0.99)
    loss = tf.reduce_mean(tf.math.reduce_sum(-(y * tf.math.log(y_hat)), axis=1))
    return loss


def cee(y, y_hat):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))


def train(model, x, y, optimizer):
    with tf.GradientTape() as t:
        current_loss = crossentropy(y=y, y_hat=model(x))
        # current_loss = cee(y, y_hat=model(x))
    deltas = t.gradient(current_loss, model.trainable_variables)
    lr = 0.01
    # [var.assign_sub(lr * delta) for var, delta in zip(model.trainable_variables, deltas)]
    optimizer.apply_gradients(zip(deltas, model.trainable_variables))

    return current_loss


if __name__ == '__main__':
    (train_xs, train_ys), (test_xs, test_ys) = load_data()
    train_xs = (train_xs.reshape(-1, 784) / 255.).astype(np.float32)
    train_ys = to_categorical(train_ys, 10)

    batch_xs = train_xs[:4]
    batch_ys = train_ys[:4]

    # define model
    simple_dnn = SimpleDNN(name='simple_dnn')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for i in range(10):
        loss = train(simple_dnn, batch_xs, batch_ys, optimizer)
        print('loss: {}'.format(loss))
