from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Reshape
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model

encoder_input = Input(shape=(28, 28, 1), name="img")
x = Conv2D(16, 3, activation="relu")(encoder_input)
x = Conv2D(32, 3, activation="relu")(x)
x = MaxPooling2D(3)(x)
x = Conv2D(32, 3, activation="relu")(x)
x = Conv2D(16, 3, activation="relu")(x)
encoder_output = GlobalMaxPooling2D()(x)

encoder = Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

x = Reshape((4, 4, 1))(encoder_output)
x = Conv2DTranspose(16, 3, activation="relu")(x)
x = Conv2DTranspose(32, 3, activation="relu")(x)
x = UpSampling2D(3)(x)
x = Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder = Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
