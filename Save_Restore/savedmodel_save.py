import tensorflow as tf


class Dense(tf.Module):
    def __init__(self, in_features, out_features, activation=None, name=None):
        self.activation = activation
        super().__init__(name=name)
        with self.name_scope:
            self.w = tf.Variable(tf.random.normal([in_features, out_features], stddev=0.1), name='w')
            self.b = tf.Variable(tf.zeros([out_features]), name='b')

    @tf.function
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


if __name__ == '__main__':
    batch_xs = [[0.1, 0.2]]
    dense = Dense(2, 2, 'relu')
    dense(batch_xs)
    tf.saved_model.save(dense, 'model')
