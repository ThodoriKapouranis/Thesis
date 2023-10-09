from dataclasses import dataclass
import os
import logging
import matplotlib
from absl import app, flags
import numpy as np
import tensorflow as tf
from keras.metrics import MeanIoU

from config import validate_config
from DatasetHelpers.Dataset import convert_to_tfds, create_dataset

from Models.XGB import Batched_XGBoost
from Models.UNet import UNetCompiled
script_path = os.path.dirname(os.path.realpath(__file__))

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

# matplotlib.style.use("classic")
matplotlib.rc("font", **font)

FLAGS = flags.FLAGS

flags.DEFINE_bool("debug", False, "Set logging level to debug")
flags.DEFINE_integer("scenario", 1, "Training data scenario. \n\t 1: Only co_event \n\t 2: coevent & preevent \n\t 3: coevent & preevent & coherence")
flags.DEFINE_string('s1_co', '/workspaces/Thesis/10m_data/s1_co_event_grd', 'filepath of Sentinel-1 coevent data')
flags.DEFINE_string('s1_pre', '/workspaces/Thesis/10m_data/s1_pre_event_grd', 'filepath of Sentinel-1 prevent data')
flags.DEFINE_string('s2_weak', '/workspaces/Thesis/10m_data/s2_labels', 'filepath of S2-weak labelled data')
flags.DEFINE_string('coh_co', '/workspaces/Thesis/10m_data/coherence/co_event', 'filepath of coherence coevent data')
flags.DEFINE_string('coh_pre', '/workspaces/Thesis/10m_data/coherence/pre_event', 'filepath of coherence prevent data')

flags.DEFINE_string('hand_coh_co', '/workspaces/Thesis/10m_hand/coherence_10m/hand_labeled/co_event', '(h) filepath of coevent data')
flags.DEFINE_string('hand_coh_pre', '/workspaces/Thesis/10m_hand/coherence_10m/hand_labeled/pre_event', '(h) filepath of preevent data')
flags.DEFINE_string('hand_s1_co', '/workspaces/Thesis/10m_hand/HandLabeled/S1Hand', '(h) filepath of Sentinel-1 coevent data')
flags.DEFINE_string('hand_s1_pre', '/workspaces/Thesis/10m_hand/S1_Pre_Event_GRD_Hand_Labeled', '(h) filepath of Sentinel-1 prevent data')
flags.DEFINE_string('hand_labels', '/workspaces/Thesis/10m_hand/HandLabeled/LabelHand', 'filepath of hand labelled data')

# Model specific flags
flags.DEFINE_string("model", None, "'xgboost', 'unet', 'transunet', 'segformer'")
flags.DEFINE_bool("baseline", False, "T/F for baseline. If true, it does not apply the new processing pipeline")

# XGB boost specific parameters
flags.DEFINE_integer('xgb_batches', 4, 'batches to use for splitting xgboost training to fit in memory')

# NN training Hyperparameters
flags.DEFINE_integer("batch_size", 1, "Batch size to use for training")
flags.DEFINE_integer("epochs", 5, "Number of epochs to train model for")
flags.DEFINE_float("lr", 1e-4, "Defines starting learning rate")
flags.DEFINE_integer("embedding_size", 768, "Embedding (hidden) layer to use for transunet model")

# Transunet specific parameters
flags.DEFINE_integer("patch_size", 16, "Patch size to use for transformer (ViT) model")
flags.DEFINE_list("decoder_channels", None, "Custom decoder channels to use for Decoder Cup stage (list of strings)")


# Define model metadata
flags.DEFINE_string("savename", None, "Name to use to save the model")

def main(x):
    validate_config(FLAGS)
    if FLAGS.scenario == 3:
        channel_size = 6
    elif FLAGS.scenario == 2:
        channel_size = 4
    else:
        channel_size = 2
    
    # XGboost uses a different kind of dataloader than the Tensorflow models.
    if FLAGS.model == 'xgboost':
        xgb = Batched_XGBoost()
        dataset = create_dataset(FLAGS)
        batches = dataset.generate_batches(FLAGS.xgb_batches)
        xgb.train_in_batches(batches, skip_missing_data=False)
        xgb.model.save_model(f"Results/Models/{FLAGS.savename}.json")

        
    
    else:
        # Generic tensorflow NN hyperparameter and dataset creation
        model=None
        dataset = create_dataset(FLAGS)
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            FLAGS.lr,
            decay_steps=200,
            decay_rate=0.96,
            staircase=True
        )
        
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam',
        )

        if FLAGS.model == 'unet':
            train_ds, val_ds, test_ds, hand_ds = convert_to_tfds(dataset, channel_size, 'HWC', baseline=FLAGS.baseline)
            BATCH_SIZE = FLAGS.batch_size 
            # Set up datasets (Set batch size or else everything will break)
            train_ds = (
                train_ds
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE)
            )

            val_ds = (
                val_ds
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE)
            )


            model = UNetCompiled(input_size=(512, 512, channel_size), n_filters=64, n_classes=2)
            print(model.summary())
            
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=opt,
                weighted_metrics=[],
                metrics=[MeanIoU(num_classes=2, sparse_y_pred=False)]
            )

        if FLAGS.model == "transunet":
            train_ds, val_ds, test_ds, hand_ds = convert_to_tfds(dataset, channel_size, 'HWC')
            for img, tgt, wgt in train_ds.take(1):
                print(img.shape, tgt.shape, wgt.shape)


            grid_size = (512 // FLAGS.patch_size, 512 // FLAGS.patch_size )
            # Depending on our grid size our decoder structure will need to have more Conv2dRelu + upscaling layers to get back to the original 512x512 size.
            decoder_channels = FLAGS.decoder_channels

            if decoder_channels == None:
                # Generate decoder channels that accomodate the patch size to ensure image gets upscaled back to original resolution
                decoderblock_amount = int(np.log2( 512 //  grid_size[0]))
                decoder_channels = [ 16 * 2**x for x in reversed(range(decoderblock_amount)) ]
            else:
                # Ensure that list objects are ints
                decoder_channels = [int(x) for x in decoder_channels]

            print(grid_size)
            print(decoder_channels)

            model = TransUNet(
                image_size=512,
                hidden_size=FLAGS.embedding_size,
                channels=channel_size, 
                patch_size=FLAGS.patch_size, 
                grid=grid_size,
                decoder_channels=decoder_channels,
                num_classes=2, 
                hybrid=False, 
                pretrain=False
            )
            print(model.summary())
            
            # Logits false bc thats what the transunet github uses and I dont want to mess with it
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                optimizer=opt,
                metrics=[MeanIoU(num_classes=2, sparse_y_pred=False)]
            )

        if FLAGS.model == 'segformer':
            train_ds, val_ds, test_ds, hand_ds = convert_to_tfds(dataset, channel_size, 'CHW')
            BATCH_SIZE = FLAGS.batch_size
            
            train_ds = (
                train_ds
                .cache()
                .shuffle(BATCH_SIZE * 10)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE)
            )
            val_ds = (
                val_ds
                .cache()
                .shuffle(BATCH_SIZE * 10)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE)
            )

            print(train_ds.element_spec)
            for img, tgt, wgt in train_ds.take(1):
                print(img.shape, tgt.shape, wgt.shape)

            # Huggingface models require datasets to be in Channel first format.
            segformer_config = SegformerConfig(
                num_channels = channel_size,
                # depths= [ 3,6,40,3 ], # MiT-b5,
                # hidden_sizes = [64, 128, 320, 512], # MiT-b5
                # decoder_hidden_size= 768 #MiT-b5
            )
            model = TFSegformerForSemanticSegmentation(segformer_config)
            model.build( (BATCH_SIZE, channel_size, 512, 512) )


            opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
            model.compile(
                # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                optimizer=opt,
                weighted_metrics=[]
                # metrics=[MeanIoU(num_classes=2, sparse_y_pred=False)]
            )
        
        CLASS_W = {0: 0.6212519560516805, 1: 2.5618224079902174}  # Empirical 
        results = model.fit(train_ds, epochs=FLAGS.epochs, validation_data=val_ds, validation_steps=32)
        
        if FLAGS.model == "segformer":
            model.save_pretrained(f"Results/Models/{FLAGS.savename}")
            
        else:
            model.save(f"Results/Models/{FLAGS.savename}")

if __name__ == "__main__":
    app.run(main)