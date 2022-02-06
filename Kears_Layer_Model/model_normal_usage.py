from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input_ = Input(shape=[784])
layer1 = Dense(units=256, activation='relu')(input_)
layer2 = Dense(units=256, activation='relu')(layer1)
output = Dense(units=10, activation='softmax')(layer1)

dnn = Model(input_, output)
