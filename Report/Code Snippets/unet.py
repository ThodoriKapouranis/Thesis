from dataclasses import dataclass, field
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, Dense, concatenate, Dropout, BatchNormalization, Activation, Flatten, RandomFlip, RandomRotation
from absl import flags, app
from keras.regularizers import L2
from keras.metrics import MeanIoU

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = Conv2D(n_filters, 
        3,  # filter size
        activation='relu',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(),
        kernel_initializer='HeNormal')(inputs)

    conv = Conv2D(n_filters, 
        3,  # filter size
        activation='relu',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(),
        kernel_initializer='HeNormal')(conv)

    conv = BatchNormalization()(conv, training=False)
    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv
    skip_connection = conv    
    return next_layer, skip_connection

# 2x2 up-conv, merge with skip connection, 3x3 conv, 3x3 conv
def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    up = Conv2DTranspose(
                n_filters,
                (3,3),
                strides=(2,2),
                padding='same')(prev_layer_input)

    merge = concatenate([up, skip_layer_input], axis=3)

    conv = Conv2D(n_filters, 
                3,  
                activation='relu',
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(),
                kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                3, 
                activation='relu',
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(),
                kernel_initializer='HeNormal')(conv)
    return conv

# Assemble the full model
def UNetCompiled(input_size=(512, 512, 2), n_filters=32, n_classes=2):

    # Input size represent the size of 1 image (the size used for pre-processing) 
    inputs = Input(input_size)
    
    # Data augmentation layers
    inputs = RandomFlip()(inputs)
    inputs = RandomRotation(0.2)(inputs)
    
    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image 
    cblock1 = EncoderMiniBlock(inputs, n_filters,dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 
    
    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers) 
    # Followed by a 1x1 Conv layer to get the image to the desired size. 
    # Observe the number of channels will be equal to number of output classes 
    conv9 = Conv2D(n_filters,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(ublock9)

    out = Conv2D(n_classes, 1, padding='same')(conv9)
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model
