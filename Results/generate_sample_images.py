from absl import app, flags
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transunet import TransUNet
import sys
sys.path.append('../Thesis')
from DatasetHelpers.Dataset import create_dataset, convert_to_tfds

FLAGS = flags.FLAGS
flags.DEFINE_bool("debug", False, "Set logging level to debug")
flags.DEFINE_integer("scenario", 1, "Training data scenario. \n\t 1: Only co_event \n\t 2: coevent & preevent \n\t 3: coevent & preevent & coherence")
flags.DEFINE_string("model_path", "/workspaces/Thesis/Results/Models/unet_scenario1_64", "'xgboost', 'unet', 'a-unet")
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


correct_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "blue"])
missing_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "lightgrey"])
wrong_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "magenta"])
def main(x):
    model = tf.keras.models.load_model(FLAGS.model_path)
    model_name = FLAGS.model_path.split('/')[-1]
    if model_name == '': # Look back one more slash
        model_name = FLAGS.model_path.split('/')[-2]
    
    try:
        os.mkdir(f"/workspaces/Thesis/Results/Sample images/{model_name}/")
    except:
        pass

    # print(model.summary())
    dataset = create_dataset(FLAGS)
    channels = 2
    if FLAGS.scenario == 2:
        channels = 4
    elif FLAGS.scenario == 3:
        channels = 6

    _, _, holdout_set, hand_set = convert_to_tfds(dataset, channels)

    for i, scene in enumerate( list(hand_set.take(5)) ):
        img, tgt, _ =  scene
        
        logits = model(img)
        pred = tf.argmax(logits, axis=3)
        tgt = tgt[0, :, :, 0] # Remove extra dim
        pred = pred[0, :, :] # Remove extra dim
        
        FN_mask = np.ma.masked_where(pred==1, pred)
        FP_mask = np.ma.masked_where(tgt==1, pred)

        f, ax = plt.subplots(1,2)
        f.set_figwidth(5)
        f.set_figheight(5)

        ax[0].imshow(tgt, cmap=correct_cmap, interpolation='none')
        
        # Layer prediction image
        ax[1].imshow(pred, cmap=correct_cmap, interpolation='none')
        ax[1].imshow(np.ma.masked_array(tgt, FN_mask), cmap=missing_cmap, interpolation='none') # <--- Ground truth as gray, to show missed spots
        ax[1].imshow(FP_mask, cmap=wrong_cmap, interpolation='none')

        f.savefig(f"Results/Sample images/{model_name}/{i}")

if __name__ == "__main__":
    app.run(main)