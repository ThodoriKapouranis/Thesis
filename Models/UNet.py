from dataclasses import dataclass, field
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Input
# Could use https://github.com/yingkaisha/keras-unet-collection

from keras_unet_collection import layer_utils

class UNetPipeline:
    '''
    Dataloader meant to be used for the UNet model.
    Converts the raw filenames dataset created by DatasetHelpers.create_dataset() to usable data.
    '''
    def load_data(self, X, Y):
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
    def __init__(self, filters:int, kernel_size:tuple, pool_size:tuple, name="UNet-Encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer1 = Conv2D(filters=filters, kernel_size=kernel_size, activation='ReLU')
        self.layer2 = Conv2D(filters=filters, kernel_size=kernel_size, activation='ReLU')
        self.layer3 = MaxPooling2D(pool_size=pool_size) if pool_size else None

    def call(self, inputs):
        inputs = self.layer1(inputs)
        inputs = self.layer2(inputs)
        if self.layer3:
            inputs = self.layer3(inputs)
        return inputs



def main():
    input = Input( (572,572,3) )

    x = UNetEncoderBlock(name='EncBlk-1', filters=64, kernel_size=(3,3), pool_size=(2,2))(input)
    x = UNetEncoderBlock(name='EncBlk-2', filters=64, kernel_size=(3,3), pool_size=(2,2))(x)
    x = UNetEncoderBlock(name='EncBlk-3', filters=64, kernel_size=(3,3), pool_size=(2,2))(x)
    x = UNetEncoderBlock(name='EncBlk-4', filters=64, kernel_size=(3,3), pool_size=(2,2))(x)
    x = UNetEncoderBlock(name='EncBlk-5', filters=64, kernel_size=(3,3), pool_size=None)(x)



    model = tf.keras.Model(inputs=input, outputs=x, name="UNet (just encoder actually)")
    print(model.summary())

main()