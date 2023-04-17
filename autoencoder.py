import tensorflow as tf
import numpy as np
import os
import pickle
from PIL import Image
from keras.optimizers import Adam


class Autoencoder:
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self.model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, lr=0.01):
        optimizer = Adam(learning_rate=lr)
        mse = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse, metrics=['categorical_accuracy'])

    def train(self, x_train, batch_size, epoch):
        callback = tf.keras.callbacks.TensorBoard(log_dir="Log/")
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=epoch,
                       shuffle=True,
                       callbacks=[callback])

    def save(self, path="."):
        self._create_folder(path)
        self._save_parameters(path)
        self._save_weights(path)

    def _create_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _save_parameters(self, path):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim]
        save_path = os.path.join(path, "parameters.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, path):
        save_path = os.path.join(path, "weights.h5")
        self.model.save_weights(save_path)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls, path="."):
        parameters_path = os.path.join(path, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(path, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self.model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = tf.keras.Model(model_input, model_output)

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.model_input = encoder_input
        self.encoder = tf.keras.Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return tf.keras.Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        for layer_index in range(self._num_conv_layers):
            encoder_input = self._add_conv_layer(layer_index, encoder_input)
        return encoder_input

    def _add_conv_layer(self, layer_index, x):
        layer_number = layer_index + 1
        conv_layer = tf.keras.layers.Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = tf.keras.layers.ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        self._shape_before_bottleneck = tf.keras.backend.int_shape(x)[1:]
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.latent_space_dim, name="encoder_output")(x)
        # x = tf.keras.layers.BatchNormalization(name="encoder_output_bn")(x)
        return x

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = tf.keras.Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return tf.keras.Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_of_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = tf.keras.layers.Dense(num_of_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return tf.keras.layers.Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, reshape_layer):
        for layer_index in reversed(range(1, self._num_conv_layers)):
            reshape_layer = self._add_conv_transpose_layer(reshape_layer, layer_index)
        return reshape_layer

    def _add_conv_transpose_layer(self, x, layer_index):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = tf.keras.layers.Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = tf.keras.layers.ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"decoder_BN_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = tf.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = tf.keras.layers.Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def create_new_img(self, point, path):
        point = np.asarray([point])
        new_data = self.decoder(point)
        new_data = np.asarray(new_data[0] * 255)

        new_img = Image.fromarray(np.uint8(new_data))

        path = os.path.join(path, f"latent_space{point[0][0]}.bmp")
        new_img.save(path)

    def create_new_imgs(self, path, points):
        self._create_folder(path)
        for point in points:
            self.create_new_img(point, path)
