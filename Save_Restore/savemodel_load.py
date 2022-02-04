import tensorflow as tf

if __name__ == '__main__':
    batch_xs = [[0.1, 0.2]]
    new_model = tf.saved_model.load('model')
    new_model(batch_xs)
    print()
