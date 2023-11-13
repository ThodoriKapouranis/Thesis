from absl import app, flags
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transunet import TransUNet
import sys

from DatasetHelpers.Preprocessing import PyRAT_rlf, PyRAT_sigma, box_filter, fast_frost_filter, lee_filter
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
    model = tf.keras.Model()
    architecture = FLAGS.model_path.split('/')[-2].split('-')[0]
    if architecture == "segformer":
        ... # Not using segformer anymore
        # model = TFSegformerForSemanticSegmentation.from_pretrained(FLAGS.model_path)
    else:
        model = tf.keras.models.load_model(FLAGS.model_path)

    model_name= FLAGS.model_path.split('/')[-1]
    if model_name == '': # Look back one more slash
        model_name = FLAGS.model_path.split('/')[-2]
    
    print(model.summary())
    
    dataset = create_dataset(FLAGS)
    channels = 2
    if FLAGS.scenario == 2:
        channels = 4
    elif FLAGS.scenario == 3:
        channels = 6
    
    # Create folder ready for outputs
    try:
        os.mkdir(f"/workspaces/Thesis/Results/Sample images/{model_name}/")
    except:
        pass

    # Prepare Filters for each type 
    def identity(x):
        return x     
    
    filter = identity
    if FLAGS.filter == 'lee':
        filter = lee_filter
    if FLAGS.filter == 'box':
        filter = box_filter
    if FLAGS.filter == 'rfl':
        filter = PyRAT_rlf
    if FLAGS.filter == 'sigma':
        filter = PyRAT_sigma
    if FLAGS.filter == "frost":
        filter = fast_frost_filter
        
    print(filter)


    ds_format = "CHW" if architecture == "segformer" else "HWC"

    _, _, holdout_set, hand_set = convert_to_tfds(dataset, channels, ds_format)
    hand_set = hand_set.batch(1)
    holdout_set = holdout_set.batch(1)

    for i, scene in enumerate( list(hand_set.take(5)) ):
        img, tgt, _ =  scene
    
        # Huggingface models have a wrapper around the output. Need to access the logits.
        logits = None
        pred = None
        
        print(architecture)
        if architecture == "segformer":
            logits = model.predict(img).logits
            print(logits.shape)
            pred = tf.argmax(logits, axis=1) # BCHW, outputs at factor of (1/4, 1/4)
            pred = np.transpose(pred, axes=[1,2,0]) # Convert to HWC
            tgt = np.transpose(tgt, axes=[1,2,0]) # Convert to HWC
            pred = tf.image.resize(pred, size=(512,512))

        else:
            logits = model(img)
            print(logits.shape)
            pred = tf.argmax(logits, axis=3) # BHWC

        FN_mask = np.ma.masked_where(pred==1, pred)
        FP_mask = np.ma.masked_where(tgt==1, pred)
        f, ax = plt.subplots(1,2)
        f.set_figwidth(5)
        f.set_figheight(5)

        ax[0].imshow(tgt[0], cmap=correct_cmap, interpolation='none')
        
        # Layer prediction image
        ax[1].imshow(pred[0], cmap=correct_cmap, interpolation='none')
        ax[1].imshow(np.ma.masked_array(tgt[0], FN_mask[0]), cmap=missing_cmap, interpolation='none') # <--- Ground truth as gray, to show missed spots
        ax[1].imshow(FP_mask[0], cmap=wrong_cmap, interpolation='none')

        f.savefig(f"Results/Sample images/{model_name}/{i}")

if __name__ == "__main__":
    app.run(main)