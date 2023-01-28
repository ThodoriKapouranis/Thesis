from dataclasses import dataclass, field
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, Dense, concatenate
# Could use https://github.com/yingkaisha/keras-unet-collection

from keras_unet_collection import layer_utils

class TensorFlowDatasetConverter:
    '''
    Dataloader meant to be used for the UNet model.
    Converts the raw filenames dataset created by DatasetHelpers.create_dataset() to usable data.
    '''
    def load_data(X, Y):
        ...

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
    loader: UNetPipeline = field(init=False)


    def __init__(self):
        self.loader = UNetPipeline()

    def build(self):
        ...

    def fit(self, X, Y):
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

    def __init__(self, filters:int, kernel_size:tuple, pool_size:tuple, name="UNet-Encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='ReLU')
        self.layer2 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='ReLU')
        self.layer3 = MaxPooling2D(pool_size=pool_size) if pool_size else None

    def call(self, inputs):
        skip = self.layer1(inputs)
        skip = self.layer2(skip) 
        out = skip 
        
        if self.layer3: # apply max pooling
            out = self.layer3(out)
        
        return out, skip

class UNetDecoderBlock(tf.keras.layers.Layer):
    # UNet Decoder block to be used with keras layers functional api
    #! Need to add skip connection input
    def __init__(self, skip:any, filters:int, kernel_size:tuple, pool_size:tuple, name="UNet-Decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.skip = skip
        
        self.up = Conv2DTranspose(filters, kernel_size=kernel_size, strides=pool_size, padding='same', activation='ReLU')
        self.conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='ReLU')
        self.conv2 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='ReLU')

    def call(self, inputs):
        X = self.up(inputs)
        merge = concatenate([X, self.skip], axis=3)
        print(X.shape, self.skip.shape, merge.shape)
        print(type(X), type(self.skip), type(merge))

        X = self.conv1(merge)
        X = self.conv2(X)
        return X

def main():
    tf.compat.v1.disable_eager_execution()

    input = Input( (512,512,3) )

    enc1, skip1 = UNetEncoderBlock(name='EncBlk-1', filters=64, kernel_size=(3,3), pool_size=(2,2))(input)
    enc2, skip2 = UNetEncoderBlock(name='EncBlk-2', filters=128, kernel_size=(3,3), pool_size=(2,2))(enc1)
    enc3, skip3 = UNetEncoderBlock(name='EncBlk-3', filters=256, kernel_size=(3,3), pool_size=(2,2))(enc2)
    enc4, skip4 = UNetEncoderBlock(name='EncBlk-4', filters=512, kernel_size=(3,3), pool_size=(2,2))(enc3)

    enc5, _ = UNetEncoderBlock(name='EncBlk-5', filters=1024, kernel_size=(3,3), pool_size=None)(enc4) # No further pooling, bottom of "U"

    dec1 = UNetDecoderBlock(name='DecBlk-1', skip=skip4, filters=512, kernel_size=(3,3), pool_size=(2,2))(enc5)
    dec2 = UNetDecoderBlock(name='DecBlk-2', skip=skip3, filters=256, kernel_size=(3,3), pool_size=(2,2))(dec1)
    dec3 = UNetDecoderBlock(name='DecBlk-3', skip=skip2, filters=128, kernel_size=(3,3), pool_size=(2,2))(dec2)
    dec4 = UNetDecoderBlock(name='DecBlk-4', skip=skip1, filters=64, kernel_size=(3,3), pool_size=(2,2))(dec3)

    # Segmentation layer
    classes = 2
    out = Conv2D(name='Dense', filters=classes, kernel_size=(1,1))(dec4)

    model = tf.keras.Model(inputs=input, outputs=out, name="UNet (just encoder actually)")
    print(model.summary())

main()