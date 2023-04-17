import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pickle
from PIL import Image

from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, BatchNormalization, Lambda, ReLU, \
    Activation, Reshape
from keras import Input, Model

import keras.backend as K
tf.compat.v1.disable_eager_execution()


class Loss(tf.keras.losses.Loss):
    def __init__(self, log_var, mean, reconstruction_loss_weight):
        super().__init__()
        self.log_var = log_var
        self.mean = mean
        self.reconstruction_loss_weight = reconstruction_loss_weight

    def call(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mean) - K.exp(self.log_var), axis=1)
        return kl_loss


class VAE:
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim,
                 reconstruction_loss_weight=100):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = reconstruction_loss_weight

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
        loss = Loss(self.log_var, self.mean, self.reconstruction_loss_weight)
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, x_train, batch_size, epoch):
        callback = tf.keras.callbacks.TensorBoard(log_dir="Log/")
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=epoch,
                       shuffle=True,
                       callbacks=[callback])

    def save(self, path: os.path = "."):
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
    def load(cls, path: os.path = "."):
        parameters_path = os.path.join(path, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        vae = VAE(*parameters)
        weights_path = os.path.join(path, "weights.h5")
        vae.load_weights(weights_path)
        return vae

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
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        self._shape_before_bottleneck = tf.keras.backend.int_shape(x)[1:]
        x = Flatten()(x)

        self.mean = Dense(self.latent_space_dim, name="mean_output")(x)
        self.log_var = Dense(self.latent_space_dim, name="log_var_output")(x)

        def noice(args):
            mean, log_var = args
            d = K.random_normal(shape=(self.latent_space_dim, ))
            return K.exp(log_var/2) * d + mean

        x = Lambda(noice, output_shape=(self.latent_space_dim, ))([self.mean, self.log_var])

        return x

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_of_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_of_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, reshape_layer):
        for layer_index in reversed(range(1, self._num_conv_layers)):
            reshape_layer = self._add_conv_transpose_layer(reshape_layer, layer_index)
        return reshape_layer

    def _add_conv_transpose_layer(self, x, layer_index):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_BN_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=3,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def create_new_img(self, point, path):
        point = np.asarray([point])
        new_data = self.decoder.predict(point)
        new_data = np.asarray(new_data[0] * 255)

        new_img = Image.fromarray(np.uint8(new_data))

        path = os.path.join(path, f"latent_space{point[0][0]}.bmp")
        new_img.save(path)

    def create_new_imgs(self, path, points):
        self._create_folder(path)
        for point in points:
            self.create_new_img(point, path)

    def show_latent_space(self, *args):
        for data in args:
            points = self.encoder.predict(data)
            print(self.log_var)
            plt.scatter(x=points[:, 0], y=points[:, 1])

        plt.show()
