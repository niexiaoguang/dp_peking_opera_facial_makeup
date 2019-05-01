import glob
import io
import math
import time
import os

import keras.backend as K
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Sequential, Input, Model
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from scipy.misc import imread, imsave
from scipy.stats import entropy

# K.set_image_dim_ordering('tf')

np.random.seed(1337)

def write_log(callback, name, loss, batch_no):
    """
    Write training summary to TensorBoard
    """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = loss
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()

    
    
def build_generator():
    gen_model = Sequential()

    gen_model.add(Dense(input_dim=100, output_dim=2048))
    gen_model.add(LeakyReLU(alpha=0.2))



    gen_model.add(Dense(256 * 8 * 8))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU(alpha=0.2))


    # 8x8 
    gen_model.add(Reshape((8, 8, 256), input_shape=(256 * 8 * 8,)))
    gen_model.add(UpSampling2D(size=(2, 2)))

    # 16x16
    gen_model.add(Conv2D(128, (5, 5), padding='same'))
    gen_model.add(LeakyReLU(alpha=0.2))

    gen_model.add(UpSampling2D(size=(2, 2)))

    # 32x32
    gen_model.add(Conv2D(64, (5, 5), padding='same'))
    gen_model.add(LeakyReLU(alpha=0.2))

    gen_model.add(UpSampling2D(size=(2, 2)))


    # 64x64
    gen_model.add(Conv2D(32, (5, 5), padding='same'))
    gen_model.add(LeakyReLU(alpha=0.2))

#     gen_model.add(UpSampling2D(size=(2, 2)))


#     # 128x128
    gen_model.add(Conv2D(3, (5, 5), padding='same'))
    gen_model.add(LeakyReLU(alpha=0.2))


    return gen_model


def build_discriminator():


    dis_model = Sequential()

    w = 64
    h = 64
    dis_model.add(
        Conv2D(128, (5, 5),
               padding='same',
               input_shape=(w, h, 3))
    )


    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    # 32x32
    dis_model.add(Conv2D(256, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    # 16x16
    dis_model.add(Conv2D(512, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    # 8x8
    dis_model.add(Flatten())
    dis_model.add(Dense(1024))
    dis_model.add(LeakyReLU(alpha=0.2))

    dis_model.add(Dense(1))
    dis_model.add(Activation('sigmoid'))

    return dis_model



# ===============================================================

# ===============================================================

rawImagesPath = "rawimages"
readyImagesPath = "../data/readyimages64x64"

# ===============================================================

# ===============================================================



def pre_process_img(input_path, output_path):

    w = 64
    h = 64
    files = os.listdir(rawImagesPath)
    os.chdir(input_path)
    if(not os.path.exists(output_path)):
        os.makedirs(output_path)
    for file in files:
        if(os.path.isfile(file) & file.endswith('.jpg')):
            img = Image.open(file)
            img = img.resize((w, h), Image.ANTIALIAS)
            img.save(os.path.join(readyImagesPath,file))



def post_process_img(img):
    res = img
    return res

def train():
    batch_size = 64
    z_shape = 100
    epochs = 10000
    dis_learning_rate = 0.0005
    gen_learning_rate = 0.0004
    dis_momentum = 0.9
    gen_momentum = 0.9
    dis_nesterov = True
    gen_nesterov = True
    
    timestamp = int(time.time())
    res_path = "../data/results/" + str(timestamp) + "/"
    
    if not os.path.exists(res_path):
        os.mkdir(res_path)



    # Loading images
    all_images = []
    for index, filename in enumerate(glob.glob(readyImagesPath + '/*.jpg')): 
        image = imread(filename, flatten=False, mode='RGB')
        all_images.append(image)

    # Convert to Numpy ndarray
    X = np.array(all_images)
    X = (X - 127.5) / 127.5


    # Define optimizers
    dis_optimizer = SGD(lr=dis_learning_rate, momentum=dis_momentum, nesterov=dis_nesterov)
    gen_optimizer = SGD(lr=gen_learning_rate, momentum=gen_momentum, nesterov=gen_nesterov)


    gen_model = build_generator()
    gen_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)


    dis_model = build_discriminator()
    dis_model.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    
    
        
    
    adversarial_model = Sequential()
    adversarial_model.add(gen_model)
    dis_model.trainable = False
    adversarial_model.add(dis_model)


    adversarial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), write_images=True, write_grads=True,
                                  write_graph=True)

    tensorboard.set_model(gen_model)
    tensorboard.set_model(dis_model)


    for epoch in range(epochs):
        print("Epoch is", epoch)
        number_of_batches = int(X.shape[0] / batch_size)
        print("Number of batches", number_of_batches)
        for index in range(number_of_batches):


            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))


            image_batch = X[index * batch_size:(index + 1) * batch_size]

            generated_images = gen_model.predict_on_batch(z_noise)

            y_real = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            y_fake = np.random.random_sample(batch_size) * 0.2

            dis_loss_real = dis_model.train_on_batch(image_batch, y_real)
            dis_loss_fake = dis_model.train_on_batch(generated_images, y_fake)
            d_loss = (dis_loss_real+dis_loss_fake)/2
            print("d_loss:", d_loss)


            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            g_loss = adversarial_model.train_on_batch(z_noise, [1] * batch_size)


            print("g_loss:", g_loss)

            """
            Save losses to Tensorboard after each epoch
            """
            write_log(tensorboard, 'discriminator_loss', np.mean(d_loss), epoch)
            write_log(tensorboard, 'generator_loss', np.mean(g_loss), epoch)
            
            
            
            
            
            if epoch % 10 == 0:
                
#                 if epoch % 100 == 1:
#                     path = "/Volumes/LaMer/dl/modelsbak/facial_design_of_peking_opera/"
#                     #save models
#                     # Specify the path for the generator model
#                     gen_model.save(path + "gen_model_" + str(epoch) + ".h5") 

#                     # Specify the path for the discriminator model
#                     dis_model.save(path + "dis_model_" + str(epoch) + ".h5") 


                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
                gen_images1 = gen_model.predict_on_batch(z_noise)
                index = 0
                for img in gen_images1[:3]:
#                     save_rgb_img(img, "results/one_{}.jpg".format(epoch))
                    
                    imsave(res_path + 'img_{}_'.format(epoch) + str(index) + '.jpg',img)
                    index += 1




    # Specify the path for the generator model
    gen_model.save(res_path + "gen_model.h5") 

    # Specify the path for the discriminator model
    dis_model.save(res_path + "dis_model.h5") 


    
    
def predict(number=16):
    path = "data/models/0429/"
    filename = "gen_model.h5"
    model = load_model(path + filename)
    for i in range(number):
        z_shape = 100
        batch_size = 32
        #z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

        z_noise = np.random.logistic(0, 1, size=(batch_size, z_shape))

        gen_images1 = model.predict_on_batch(z_noise)
        index = 0
        for img in gen_images1:
            imsave(path + 'predict/' + 'predict_img_{}_{}_'.format(i,index) + '.jpg',img)
            index += 1
        

