from dataclasses import dataclass, field
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, Dense, concatenate, Dropout, BatchNormalization, Activation, Flatten, RandomFlip, RandomRotation
from absl import flags, app
from keras.regularizers import L2
from keras.metrics import MeanIoU
# Could use https://github.com/yingkaisha/keras-unet-collection
# from keras_unet_collection import layer_utils


# @dataclass
# class UNet:
#     # Parameters adapated from keras_unet_collection._model_unet_2d
#     optimizer: tf.keras.optimizers.Optimizer
#     input: tf.Tensor
#     filter_num: int
#     encoder_num: int = 2
#     decoder_num: int = 2
#     kernel:tuple = (3,3)
#     activation: str = "RELU"
#     batch_norm: bool = False
#     pool:bool = True
#     unpool:bool = True
#     pool_shape:tuple = (2,2)
#     backbone:str = None
#     weights: str = "imagenet"
#     freeze_backbone: bool = True
#     freeze_batch_norm: bool = True
#     model_name: str = "unet"

#     model: any = field(init=False)


#     def __init__(self):
#         ...

#     def build(self):
#         ...

#     def fit(self, X, Y):
#         ...
#         '''
#         Arguments:
#         -- X : Dataset created by DatasetHelpers.create_dataset(). List of filenames.
#         -- Y : Dataset created by DatasetHelpers.create_dataset(). List of filenames.
#         '''
        
# class UNetEncoderBlock(tf.keras.layers.Layer):
#     # UNet Encoder block to be used with the keras layers functional api
#     # Returns tuple (output, skip)
#     #   -- output - Final output of the layer. Either includes a max pooling or not.
#     #   -- skip   - Final output of the Conv2d layers, used as skip connection in the decoder.

#     def __init__(self, filters:int, kernel_size:tuple, pool_size:tuple, dropout:float, name="UNet-Encoder", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.conv1 = Conv2D(
#             filters=filters, 
#             kernel_size=kernel_size, 
#             padding='same', 
#             activation='ReLU', 
#             kernel_regularizer=L2(),
#             kernel_initializer="HeNormal"
#         )
#         self.conv2 = Conv2D(
#             filters=filters, 
#             kernel_size=kernel_size, 
#             padding='same', 
#             activation='ReLU', 
#             kernel_regularizer=L2(),
#             kernel_initializer="HeNormal"
#         )
#         self.dropout = Dropout(dropout) if dropout else None
#         self.pool = MaxPooling2D(pool_size=pool_size) if pool_size else None

#     def call(self, inputs):
#         conv = self.conv1(inputs)
#         conv = self.conv2(conv) 
#         conv = BatchNormalization()(conv, training=False)
        
#         # Apply additional layers to the 'skip' connection
#         skip = conv 
#         if self.dropout:
#             skip = self.dropout(skip)

#         # Apply additonal layers to the convolutions to make it the final output
#         out = skip
#         if self.pool: # apply max pooling
#             out = self.pool(out)
        
#         return out, skip

# class UNetDecoderBlock(tf.keras.layers.Layer):
#     # UNet Decoder block to be used with keras layers functional api
#     #! Need to add skip connection input
#     def __init__(self, skip:any, filters:int, kernel_size:tuple, pool_size:tuple, name="UNet-Decoder", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.skip = skip
        
#         self.up = Conv2DTranspose(filters, kernel_size=kernel_size, strides=pool_size, padding='same', activation='ReLU')
#         self.conv1 = Conv2D(
#             filters=filters, 
#             kernel_size=kernel_size, 
#             padding='same', 
#             activation='ReLU', 
#             kernel_regularizer=L2(),
#             kernel_initializer="HeNormal"
#         )
#         self.conv2 = Conv2D(
#             filters=filters, 
#             kernel_size=kernel_size, 
#             padding='same', 
#             activation='ReLU', 
#             kernel_regularizer=L2(),
#             kernel_initializer="HeNormal"
#         )
        
#     def call(self, inputs):
#         X = self.up(inputs)
        
#         merge = concatenate([X, self.skip], axis=3)

#         X = self.conv1(merge)
#         X = self.conv2(X)
#         return X

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
    # inputs = RandomFlip()(inputs)
    # inputs = RandomRotation(0.2)(inputs)
    
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

def main(x):
    _test(x)

def _test(x):
    import sys
    sys.path.append('../Thesis')
    from DatasetHelpers.Dataset import create_dataset, convert_to_tfds

    FLAGS = flags.FLAGS
    flags.DEFINE_bool("debug", False, "Set logging level to debug")
    flags.DEFINE_integer("scenario", 1, "Training data scenario. \n\t 1: Only co_event \n\t 2: coevent & preevent \n\t 3: coevent & preevent & coherence")
    flags.DEFINE_string("model", "xgboost", "'xgboost', 'unet', 'a-unet")
    flags.DEFINE_string('s1_co', '/workspaces/Thesis/10m_data/s1_co_event_grd', 'filepath of Sentinel-1 coevent data')
    flags.DEFINE_string('s1_pre', '/workspaces/Thesis/10m_data/s1_pre_event_grd', 'filepath of Sentinel-1 prevent data')
    flags.DEFINE_string('hand_co', '/workspaces/Thesis/10m_data/coherence/hand_labeled/co_event', 'filepath of handlabelled coevent data')
    flags.DEFINE_string('hand_pre', '/workspaces/Thesis/10m_data/coherence/hand_labeled/pre_event', 'filepath of handlabelled preevent data')
    flags.DEFINE_string('s2_weak', '/workspaces/Thesis/10m_data/s2_labels', 'filepath of S2-weak labelled data')
    flags.DEFINE_string('coh_co', '/workspaces/Thesis/10m_data/coherence/co_event', 'filepath of coherence coevent data')
    flags.DEFINE_string('coh_pre', '/workspaces/Thesis/10m_data/coherence/pre_event', 'filepath of coherence prevent data')

    model = UNetCompiled(input_size=(512,512,2), n_filters=64, n_classes=2)
    print(model.summary())

    # Dataset initialization
    # -----------------------
    # USE_CLASS_WEIGHTS = True
    dataset = create_dataset(FLAGS)
    # 
    # CLASS_W = {0: 1.0, 1: 1.0} # Needs to be global to be used by tensorflow dataloader since it only takes img URL as input.
    # if USE_CLASS_WEIGHTS:
    #     CLASS_W = {0: 0.6212519560516805, 1: 2.5618224079902174}
    # print(f"Using class weights : {CLASS_W}")    

    # Modify the dataset to only use a tiny slice of data to overfit to test functionality
    # dataset.x_train, dataset.y_train = dataset.x_train[0:1], dataset.y_train[0:1]
    # dataset.x_val, dataset.y_val = dataset.x_train, dataset.y_train
    
    train_ds, val_ds, test_ds, = convert_to_tfds(dataset)

    # Get classweights by using only visible training data.
    DEBUG_FORCE_CLASS_WEIGHT_CALCULATION = False
    if DEBUG_FORCE_CLASS_WEIGHT_CALCULATION:
        @tf.function
        def reduce_labels(state, data) -> np.int64:
            _, y = data
            return state + tf.math.count_nonzero(y)

        print("Manually calculating BALANCED class weights ... ")
        total_pixels = len(train_ds)*512*512
        water_count = train_ds.reduce(np.int64(0), reduce_labels).numpy()
        non_water_count = total_pixels - water_count
        
        water_weight =  total_pixels / (2 * water_count )
        non_water_weight = total_pixels / (2 * non_water_count )
        class_weights = { 0 : non_water_weight, 1 : water_weight }
        print(class_weights)

    _EPOCHS = 10
    _LR = 1e-4
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(_LR,
                                                             decay_steps=200,
                                                             decay_rate=0.96,
                                                             staircase=True)
    opt = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam',
    )

    model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=opt,
            metrics=[MeanIoU(num_classes=2, sparse_y_pred=False)]
    )
    
    train_ds.shuffle(300)
    # results = model.fit(train_ds, epochs=_EPOCHS)
    results = model.fit(train_ds, epochs=_EPOCHS, validation_data=val_ds, validation_steps=32)

    model.save("Results/Models/unet_scenario1_64")
    # pred = model.predict(train_ds)  # These are logits
    # pred_classes = tf.argmax(pred, axis=3) 

    
    # print(pred.shape)
    # print(pred_classes)

if __name__ == "__main__":
    app.run(main)