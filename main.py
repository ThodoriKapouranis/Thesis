from dataclasses import dataclass
import os
import logging
import matplotlib
from absl import app, flags
import numpy as np
import tensorflow as tf
from Models.UNet import UNetCompiled

from config import validate_config
from DatasetHelpers.Dataset import convert_to_tfds, create_dataset
from Models.XGB import Batched_XGBoost
from keras.metrics import MeanIoU

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
flags.DEFINE_string('hand_co', '/workspaces/Thesis/10m_data/coherence/hand_labeled/co_event', 'filepath of handlabelled coevent data')
flags.DEFINE_string('hand_pre', '/workspaces/Thesis/10m_data/coherence/hand_labeled/pre_event', 'filepath of handlabelled preevent data')
flags.DEFINE_string('s2_weak', '/workspaces/Thesis/10m_data/s2_labels', 'filepath of S2-weak labelled data')
flags.DEFINE_string('coh_co', '/workspaces/Thesis/10m_data/coherence/co_event', 'filepath of coherence coevent data')
flags.DEFINE_string('coh_pre', '/workspaces/Thesis/10m_data/coherence/pre_event', 'filepath of coherence prevent data')

# Model specific flags
flags.DEFINE_string("model", None, "'xgboost', 'unet', 'a-unet")
flags.DEFINE_integer('xgb_batches', 4, 'batches to use for splitting xgboost training to fit in memory')

# NN training Hyperparameters
flags.DEFINE_integer("epochs", 10, "Number of epochs to train model for")
flags.DEFINE_float("lr", 1e-4, "Defines starting learning rate")

# Define model metadata
flags.DEFINE_string("savename", None, "Name to use to save the model")

def main(x):
    validate_config(FLAGS)
    
    # XGboost uses a different kind of dataloader than the Tensorflow models.
    if FLAGS.model == 'xgboost':
        model = Batched_XGBoost()
        dataset = create_dataset(FLAGS)
        batches = dataset.generate_batches(FLAGS.xgb_batches)
        model.train_in_batches(batches)
    
    else:
        # Generic tensorflow NN hyperparameter and dataset creation
        model=None
        dataset = create_dataset(FLAGS)
        train_ds, val_ds, test_ds, = convert_to_tfds(dataset)
        
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
            model = UNetCompiled(input_size=(512,512,2), n_filters=64, n_classes=2)
            print(model.summary())
            
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=opt,
                metrics=[MeanIoU(num_classes=2, sparse_y_pred=False)]
            )


        results = model.fit(train_ds, epochs=FLAGS.epochs, validation_data=val_ds, validation_steps=32)
        model.save(f"Results/Models/{FLAGS.savename}")

if __name__ == "__main__":
    app.run(main)