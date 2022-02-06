from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
import numpy as np

# multi input
input_1 = Input(shape=[784])
output_1 = Dense(units=10, activation='relu', name='output_1')(input_1)

input_2 = Input(shape=[784])
output_2 = Dense(units=10, activation='relu', name='output_2')(input_2)

# single output
inputs = (input_1, input_2)
outputs = {"output_1": output_1, "output_2": output_2}
#outputs = [output_1, output_2]
model = Model(inputs, outputs)

x = tf.zeros(shape=[1, 784])
input_values = {"input_1": x, "input_2": x}

#
#losses = ['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy']
losses = {"output_1": 'sparse_categorical_crossentropy', "output_2": 'sparse_categorical_crossentropy'}
model.compile('rmsprop', loss=losses)

batch_xs = np.zeros(shape=[1, 784], dtype=np.float32)
batch_ys = np.zeros(shape=[1], dtype=np.float32)

batch_ys_bucket = {'output_1': batch_ys, 'output_2': batch_ys}
batch_ys_bucket = [batch_ys, batch_ys]
model.fit([batch_xs, batch_xs], batch_ys_bucket)

# model 실행 : __call__() => numpy
print(model.predict(input_values))

# model 실행 : Model.predict() => tensor
print(model(input_values))
