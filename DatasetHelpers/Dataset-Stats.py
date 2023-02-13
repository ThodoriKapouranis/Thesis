
from absl import app, flags
import tensorflow as tf
import numpy as np

def main(args):
    _test(args)

@tf.function
def check_for_nan(state:dict, data):
    x, y, _ = data
    temp = state.copy() # Copy of the state, will be the new state on function return.
    temp['count'] += 1
    x_nan = tf.math.count_nonzero(tf.math.is_nan(x))
    if x_nan > 0: print(f"NaNs found in data file. {x_nan}")

    y_nan = tf.math.count_nonzero(tf.math.is_nan(y))
    if y_nan > 0: print(f"NaNs found in labels file. {y_nan}")

    temp['x'] += x_nan
    temp['y'] += y_nan
    return temp

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

    dataset = create_dataset(FLAGS)
    
    dataset.x_train = dataset.x_train[0:100]
    dataset.y_train = dataset.y_train[0:100]
    
    train_ds, test_ds, val_ds = convert_to_tfds(dataset)
    nan_count = train_ds.reduce({'count':np.int64(0), 'x':np.int64(0), 'y':np.int64(0)}, check_for_nan)
    print(nan_count)


if __name__ == "__main__":
    app.run(main)