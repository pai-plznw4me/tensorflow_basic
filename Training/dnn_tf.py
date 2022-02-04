import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

in_features = 784
out_features = 256

w1 = tf.Variable(tf.random.normal([in_features, out_features], stddev=0.1), name='w1')
b1 = tf.Variable(tf.zeros([out_features]), name='b1')

in_features = 256
out_features = 10

w2 = tf.Variable(tf.random.normal([in_features, out_features], stddev=0.1), name='w2')
b2 = tf.Variable(tf.zeros([out_features]), name='b2')


def dense(x, w, b, act):
    return act(x @ w + b)


def crossentropy(y, y_hat):
    y = tf.clip_by_value(y, 1e-7, 0.99)
    y_hat = tf.clip_by_value(y_hat, 1e-7, 0.99)
    loss = tf.reduce_mean(-tf.math.reduce_sum(y * tf.math.log(y_hat), axis=1))
    return loss


def cee(y, y_hat):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))


def update_weight(xs, ys, optimizer):
    with tf.GradientTape() as t:
        layer = dense(xs, w1, b1, tf.nn.relu)
        ys_hat = dense(layer, w2, b2, tf.identity)
        current_loss = crossentropy(ys, ys_hat)

    deltas = t.gradient(current_loss, [w1, b1, w2, b2])
    print(tf.reduce_sum(deltas[0]))
    optimizer.apply_gradients(zip(deltas, [w1, b1, w2, b2]))
    print(current_loss)


(train_xs, train_ys), (test_xs, test_ys) = load_data()
train_xs = train_xs.reshape(-1, 784) / 255.
train_ys = to_categorical(train_ys, 10)

batch_ys = train_ys[:6]
batch_xs = train_xs[:6]

optimizer = tf.keras.optimizers.Adam(learning_rate=1.0)
for i in range(5):
    update_weight(batch_xs, batch_ys, optimizer)
