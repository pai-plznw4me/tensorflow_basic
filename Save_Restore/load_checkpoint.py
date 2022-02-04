import tensorflow as tf


class Dense(tf.Module):
    def __init__(self, in_features, out_features, activation=None, name=None):
        self.activation = activation
        super().__init__(name=name)
        with self.name_scope:
            tf.random.set_seed(0)
            self.w = tf.Variable(tf.random.normal([in_features, out_features], stddev=0.1), name='w')
            self.b = tf.Variable(tf.zeros([out_features]), name='b')

    @tf.function
    def __call__(self, x):
        print(self.w)
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
    chkp_path = 'chk_point'
    dense = Dense(2, 2, 'relu')
    checkpoint = tf.train.Checkpoint(model=dense)
    checkpoint.restore(chkp_path)

    batch_xs = [[0.1, 0.2]]
    print(dense(batch_xs))



