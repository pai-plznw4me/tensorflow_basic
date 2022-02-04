import tensorflow as tf
import matplotlib.pyplot as plt


def generate_liner_dataset(w=3, b=2, n_samples=201):
    # A vector of random x values
    x = tf.linspace(-2, 2, n_samples)
    x = tf.cast(x, tf.float32)

    def f(x):
        return x * w + b

    # Generate some noise
    noise = tf.random.normal(shape=[n_samples])

    # Calculate y
    y = f(x) + noise
    return x, y


class LinearRegressor(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0, True)
        self.b = tf.Variable(2.0, True)

    def __call__(self, x):
        return self.w * x + self.b


def loss(y, y_hat):
    return tf.reduce_mean(tf.square(y - y_hat))


def train(model, x, y, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(y, model(x))
        # current_loss = tf.square(2.0 * model.w * x + model.b)
        # current_loss = loss()

    dw, db = t.gradient(current_loss, [model.w, model.b])
    print(dw, db)

    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


def report(model, loss):
    return 'w={:1.2f}\tb={:1.2f}\tloss{:2.5f}'.format(model.w.numpy(), model.b.numpy(), loss)


def training(model, x, y, epochs):
    for e in epochs:
        train(model, x, y, learning_rate=0.1)
        current_loss = loss(y, model(x))

        print('Epoch : {:2d}'.format(e))
        print('        ', report(model, current_loss))


if __name__ == '__main__':
    x_, y_ = generate_liner_dataset()
    model = LinearRegressor()

    epoches = range(10)
    training(model, x_, y_, epoches)
