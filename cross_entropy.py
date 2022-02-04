import tensorflow as tf

y_true = [[0., 1., 0.], [0., 0., 1.]]
y_pred = [[0.05, 0.95, 0.], [0.1, 0.8, 0.1]]


def crossentropy(y, y_hat):
    y = tf.clip_by_value(y, 1e-10, 0.99)
    y_hat = tf.clip_by_value(y_hat, 1e-10, 0.99)
    loss = tf.reduce_mean(-tf.math.reduce_sum(y * tf.math.log(y_hat), axis=1))
    return loss
print(crossentropy(y_true, y_pred))

cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true, y_pred).numpy())

