from PIL import Image
import os
import matplotlib.pyplot as plt
from VAE import VAE

import numpy as np

img_size = 128

rons = []

print("Load data...")
for ron_name in os.listdir("data/train_ron"):
    ron = Image.open("data/train_ron/" + ron_name).resize((img_size, img_size))
    print("Round " + ron_name + " is loaded")
    rons.append(np.asarray(ron) / 255)
    if len(rons) >= 20:
        break

print("Data is loaded")

data = np.asarray([*rons])

filters = [32, 32, 32, 64, 64, 128, 128]
kernels = [3] * len(filters)
strides = [2]
for i in range(len(filters) - 1):
    if filters[i] != filters[i + 1]:
        strides.append(2)
    else:
        strides.append(1)

print("Filters: " + str(filters))
print("Kernels: " + str(kernels))
print("Strides: " + str(strides))
"""
vae = VAE(input_shape=(img_size, img_size, 3),
          conv_filters=filters,
          conv_kernels=kernels,
          conv_strides=[2, 1,
                        2, 1,
                        2, 1],
          latent_space_dim=128,
          reconstruction_loss_weight=1000000)
"""

model_num = "42"
vae = VAE.load("models/model_" + model_num + "/model_ls_1024")
#  vae.summary()

vae.compile(lr=0.0001)
vae.train(data, batch_size=64, epoch=1)
vae.save("models/model_" + model_num + "/model_ls_" + str(vae.latent_space_dim))

vae.create_new_imgs("created_imgs/model_" + model_num + "/ls_dim_" + str(vae.latent_space_dim) + "_4",
                    np.random.random((100, vae.latent_space_dim)) * 4.0 - 2.0)

vae.show_latent_space(data)
