import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model
DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def rdn(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)
    x = Conv2D(num_filters, 3, padding='same')(x_in) 
    x = a = Conv2D(num_filters, 3, padding='same')(x) 
    for i in range(num_res_blocks):
        a = res_block(a, num_filters, res_block_scaling)
    a = Conv2D(num_filters, 3, padding='same')(a)  
    a = Conv2D(num_filters, 3, padding='same')(a)  

    x = Add()([x, a])
    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)
        
    return Model(x_in, x, name="rdn")


def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 1, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x

def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x

def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
