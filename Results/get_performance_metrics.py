from absl import app, flags
import os
import tensorflow as tf
from tensorflow import keras

'''
THIS FILE IS INTENDED TO RUN A PRETRAINED MODEL THROUGH TESTING.

To verify performance of the models, we must test them on
1. The hold out (defualt: sri-lanka) test dataset (weakly labelled)
2. The hand labelled test dataset.

Performance metrics for each test must include
1. Total mIoU
2. Water mIoU
3. Water Precision
4. Water Recall
5. Water F1
6. Nonwater mIoU
7. Nonwater Precision
8. Nonwater Recall
9. Nonwater F1
'''

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

def main(x):
    

if __name__ == "__main__":
    app.run(main)
