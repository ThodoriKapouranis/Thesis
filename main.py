from dataclasses import dataclass
import os
import logging
import matplotlib
from absl import app, flags
import numpy as np
from config import create_config

script_path = os.path.dirname(os.path.realpath(__file__))

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

# matplotlib.style.use("classic")
matplotlib.rc("font", **font)

FLAGS = flags.FLAGS
# flags.DEFINE_integer("num_features", 1, "Number of features in record")
# flags.DEFINE_integer("num_samples", 50, "Number of samples in dataset")
# flags.DEFINE_integer("batch_size", 16, "Number of samples in batch")
# flags.DEFINE_integer("num_iters", 300, "Number of SGD iterations")
# flags.DEFINE_float("learning_rate", 0.1, "Learning rate / step size for SGD")
# flags.DEFINE_integer("random_seed", 31415, "Random seed")
# flags.DEFINE_float("sigma_noise", 0.5, "Standard deviation of noise random variable")
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

def main(x):
    config = create_config(FLAGS.scenario, FLAGS.model)
    print(config)
    ...
    
if __name__ == "__main__":
    app.run(main)