
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

@tf.function
def class_count(state:dict, data):
    _, y, _ = data # ignore weights and x. We only care about labels
    temp = state.copy() # Copy of the satte, will be the new state on function return.
    
    temp['count'] += 1 # state variable for scene count

    water = tf.math.count_nonzero(y) # includes discrete labels [0,1]
    non_water = tf.math.count_nonzero(y-1) # includes discrete labels [-1,0]

    temp['0'] += non_water
    temp['1'] += water
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
    flags.DEFINE_string('s2_weak', '/workspaces/Thesis/10m_data/s2_labels', 'filepath of S2-weak labelled data')
    flags.DEFINE_string('coh_co', '/workspaces/Thesis/10m_data/coherence/co_event', 'filepath of coherence coevent data')
    flags.DEFINE_string('coh_pre', '/workspaces/Thesis/10m_data/coherence/pre_event', 'filepath of coherence prevent data')

    flags.DEFINE_string('hand_coh_co', '/workspaces/Thesis/10m_hand/coherence_10m/hand_labeled/co_event', '(h) filepath of coevent data')
    flags.DEFINE_string('hand_coh_pre', '/workspaces/Thesis/10m_hand/coherence_10m/hand_labeled/pre_event', '(h) filepath of preevent data')
    flags.DEFINE_string('hand_s1_co', '/workspaces/Thesis/10m_hand/HandLabeled/S1Hand', '(h) filepath of Sentinel-1 coevent data')
    flags.DEFINE_string('hand_s1_pre', '/workspaces/Thesis/10m_hand/S1_Pre_Event_GRD_Hand_Labeled', '(h) filepath of Sentinel-1 prevent data')
    flags.DEFINE_string('hand_labels', '/workspaces/Thesis/10m_hand/HandLabeled/LabelHand', 'filepath of hand labelled data')

    dataset = create_dataset(FLAGS)
    
    train_ds, test_ds, val_ds, hand_ds = convert_to_tfds(dataset, channel_size=2, filter=lambda x: x)

    ## Nan count for some reason i forgot why I did this
    # nan_count = train_ds.reduce({'count':np.int64(0), 'x':np.int64(0), 'y':np.int64(0)}, check_for_nan)
    # print(nan_count)

    ## Class count
    label_distribution = train_ds.reduce(
        initial_state = {'count':np.int64(0), '0':np.int64(0), '1':np.int64(0)},
        reduce_func = class_count
    )

    print(label_distribution)


if __name__ == "__main__":
    app.run(main)