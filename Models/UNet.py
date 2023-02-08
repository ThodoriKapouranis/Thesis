from dataclasses import dataclass, field
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, Dense, concatenate, Dropout, BatchNormalization, Activation, Flatten
from absl import flags, app
from keras.regularizers import L2
# Could use https://github.com/yingkaisha/keras-unet-collection
# from keras_unet_collection import layer_utils


@dataclass
class UNet:
    # Parameters adapated from keras_unet_collection._model_unet_2d
    optimizer: tf.keras.optimizers.Optimizer
    input: tf.Tensor
    filter_num: int
    encoder_num: int = 2
    decoder_num: int = 2
    kernel:tuple = (3,3)
    activation: str = "RELU"
    batch_norm: bool = False
    pool:bool = True
    unpool:bool = True
    pool_shape:tuple = (2,2)
    backbone:str = None
    weights: str = "imagenet"
    freeze_backbone: bool = True
    freeze_batch_norm: bool = True
    model_name: str = "unet"

    model: any = field(init=False)


    def __init__(self):
        ...

    def build(self):
        ...

    def fit(self, X, Y):
        ...
        '''
        Arguments:
        -- X : Dataset created by DatasetHelpers.create_dataset(). List of filenames.
        -- Y : Dataset created by DatasetHelpers.create_dataset(). List of filenames.
        '''
        
class UNetEncoderBlock(tf.keras.layers.Layer):
    # UNet Encoder block to be used with the keras layers functional api
    # Returns tuple (output, skip)
    #   -- output - Final output of the layer. Either includes a max pooling or not.
    #   -- skip   - Final output of the Conv2d layers, used as skip connection in the decoder.

    def __init__(self, filters:int, kernel_size:tuple, pool_size:tuple, dropout:float, name="UNet-Encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            padding='same', 
            activation='ReLU', 
            kernel_regularizer=L2(),
            kernel_initializer="HeNormal"
        )
        self.conv2 = Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            padding='same', 
            activation='ReLU', 
            kernel_regularizer=L2(),
            kernel_initializer="HeNormal"
        )
        self.dropout = Dropout(dropout) if dropout else None
        self.pool = MaxPooling2D(pool_size=pool_size) if pool_size else None

    def call(self, inputs):
        conv = self.conv1(inputs)
        conv = self.conv2(conv) 
        conv = BatchNormalization()(conv, training=False)
        
        # Apply additional layers to the 'skip' connection
        skip = conv 
        if self.dropout:
            skip = self.dropout(skip)

        # Apply additonal layers to the convolutions to make it the final output
        out = skip
        if self.pool: # apply max pooling
            out = self.pool(out)
        
        return out, skip

class UNetDecoderBlock(tf.keras.layers.Layer):
    # UNet Decoder block to be used with keras layers functional api
    #! Need to add skip connection input
    def __init__(self, skip:any, filters:int, kernel_size:tuple, pool_size:tuple, name="UNet-Decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.skip = skip
        
        self.up = Conv2DTranspose(filters, kernel_size=kernel_size, strides=pool_size, padding='same', activation='ReLU')
        self.conv1 = Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            padding='same', 
            activation='ReLU', 
            kernel_regularizer=L2(),
            kernel_initializer="HeNormal"
        )
        self.conv2 = Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            padding='same', 
            activation='ReLU', 
            kernel_regularizer=L2(),
            kernel_initializer="HeNormal"
        )
    def call(self, inputs):
        X = self.up(inputs)
        
        #! #! !#
        # merge = X
        merge = concatenate([X, self.skip], axis=3)
        #! #! #!

        X = self.conv1(merge)
        X = self.conv2(X)
        return X

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

    tf.compat.v1.disable_eager_execution()

    input = Input( (512,512,2) )
    f = 32
    enc1, skip1 = UNetEncoderBlock(name='EncBlk-1', filters=1*f, kernel_size=(3,3), pool_size=(2,2), dropout=None)(input)
    enc2, skip2 = UNetEncoderBlock(name='EncBlk-2', filters=2*f, kernel_size=(3,3), pool_size=(2,2), dropout=None)(enc1)
    enc3, skip3 = UNetEncoderBlock(name='EncBlk-3', filters=4*f, kernel_size=(3,3), pool_size=(2,2), dropout=0.4)(enc2)
    enc4, skip4 = UNetEncoderBlock(name='EncBlk-4', filters=8*f, kernel_size=(3,3), pool_size=(2,2), dropout=0.4)(enc3)

    enc5, _ = UNetEncoderBlock(name='EncBlk-5', filters=16*f, kernel_size=(3,3), pool_size=None, dropout=0.2)(enc4) # No further pooling, bottom of "U"

    dec1 = UNetDecoderBlock(name='DecBlk-1', skip=skip4, filters=8*f, kernel_size=(3,3), pool_size=(2,2))(enc5)
    dec2 = UNetDecoderBlock(name='DecBlk-2', skip=skip3, filters=4*f, kernel_size=(3,3), pool_size=(2,2))(dec1)
    dec3 = UNetDecoderBlock(name='DecBlk-3', skip=skip2, filters=2*f, kernel_size=(3,3), pool_size=(2,2))(dec2)
    dec4 = UNetDecoderBlock(name='DecBlk-4', skip=skip1, filters=1*f, kernel_size=(3,3), pool_size=(2,2))(dec3)

    # Segmentation layer
    classes = 1

    out = Conv2D(name='classification', filters=classes, kernel_size=(1,1), activation='sigmoid', padding='same')(dec4)

    model = tf.keras.Model(inputs=input, outputs=out, name="UNet")
    print(model.summary())

    dataset = create_dataset(FLAGS)

    # Modify the dataset to only use a tiny slice of data to overfit to test functionality
    dataset.x_train, dataset.y_train = dataset.x_train[0:1], dataset.y_train[0:1]
    dataset.x_val, dataset.y_val = dataset.x_train, dataset.y_train 
    
    train_ds, test_ds, val_ds = convert_to_tfds(dataset)

    _EPOCHS = 1000
    _LR = 1e-4
    opt = tf.keras.optimizers.Adam(
        learning_rate=_LR,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam',
    )
    model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=opt,
            metrics=['accuracy']
    )
    
    train_ds.shuffle(300)
    results = model.fit(train_ds, epochs=_EPOCHS)
    # results = model.fit(train_ds, epochs=_EPOCHS, validation_data=val_ds, validation_steps=32)

    model.save_weights("Results/Models/unet_scenario1_32")
    pred = model.predict(train_ds)
    print(pred)

if __name__ == "__main__":
    app.run(main)