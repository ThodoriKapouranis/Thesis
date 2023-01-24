from dataclasses import dataclass
import os
import logging
import matplotlib
from absl import app, flags
import numpy as np
from config import TrainingConfig, create_config
from dataset_loader import create_dataset
from models import Batched_XGBoost

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
flags.DEFINE_string("model", "xgboost", "'xgboost', 'unet', 'a-unet")
flags.DEFINE_integer('xgb_batches', 4, 'batches to use for splitting xgboost training to fit in memory')

def main(x):
    config = create_config(FLAGS.scenario, FLAGS.model)
    print(config)
    
    dataset = create_dataset(FLAGS)
    batches = dataset.generate_batches(FLAGS.xgb_batches)
    
    if config.model == 'xgboost':
        model = Batched_XGBoost()
        model.train_in_batches(batches)

    
if __name__ == "__main__":
    app.run(main)