'''
This file generated the 6x6 grids of sample image label output generation used in the Appendix of the paper.

Rows : A singular sample image
Column 0 : True labels
Column 1-5: No filter, Box, Lee, Refined Lee, Frost filter

Have 3x XGB grids, 3x UNet grids for each scenario
'''

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
from Models.XGB import Batched_XGBoost
from DatasetHelpers.Preprocessing import PyRAT_rlf, PyRAT_sigma, box_filter, fast_frost_filter, frost_filter, lee_filter
from DatasetHelpers.Dataset import create_dataset, convert_to_tfds

FLAGS = flags.FLAGS
flags.DEFINE_bool("debug", False, "Set logging level to debug")
flags.DEFINE_integer("scenario", None, "Training data scenario. \n\t 1: Only co_event \n\t 2: coevent & preevent \n\t 3: coevent & preevent & coherence")
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

flags.DEFINE_string('architecture', None, "XGB or UNet")
flags.DEFINE_string('model_paths', "Results/Models", "Path where models reside")

correct_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "lightskyblue"])
missing_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "grey"])
wrong_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "sandybrown"])

# Define our filters to be used in each column
def identity(x):
    return x 

FILTERS = [ 
    identity,
    box_filter,
    lee_filter,
    PyRAT_rlf,
    fast_frost_filter
]

FILTER_NAMES = [
    "Base",
    "Mean Filter",
    "Lee Filter",
    "RLF Filter",
    "Frost Filter"
]

# Define the models to be used.
# The scenario number should be appended to the suffix
UNET_MODELS = [
    "unet-64-scenario-",
    "cold-violet-s",
    "polished-night-s",
    "wild-lab-s",
    "baldurs-gate-s"
]

XGB_MODELS = [
    "summer-sun-s",
    "spring-glitter-s",
    "frosty-night-s",
    "floral-voice-s",
    "mute-mud-s"
]


def main(x):
    model = tf.keras.Model()
    
    FLAGS.architecture = FLAGS.architecture.lower()

    assert FLAGS.architecture in ["xgb", "unet"]
    assert FLAGS.scenario in [1, 2, 3]

    # Generate appropriate dataset
    # ------------------------------
    dataset = create_dataset(FLAGS)
    channels = 2
    if FLAGS.scenario == 2:
        channels = 4
    elif FLAGS.scenario == 3:
        channels = 6

    ## Load the dataset initially with the identity function. We will be applying the filters after we grab them.
    _, _, out_of_region, in_region = convert_to_tfds(dataset, channels, filter=identity, format = "HWC")
    out_of_region = out_of_region.batch(1).shuffle(buffer_size=5, seed=31415)
    in_region = in_region.batch(1).shuffle(buffer_size=2, seed=31415)

    # Ensure image deposit location exists
    # ------------------------------
    try:
        os.mkdir(f"/workspaces/Thesis/Results/Image Grids/{FLAGS.architecture}")
    except:
        pass

    # Grab appropriate models
    # ----------------------------
    models = [None for _ in range(5)]

    if FLAGS.architecture == "unet":
        for i, model_name in enumerate(UNET_MODELS):
            print(f"Loading {model_name}{FLAGS.scenario}")
            models[i] = tf.keras.models.load_model(f'{FLAGS.model_paths}/{model_name}{FLAGS.scenario}')
    
    if FLAGS.architecture == "xgb":
        for i, model_name in enumerate(XGB_MODELS):
            model = Batched_XGBoost()
            model.load_model(f'{FLAGS.model_paths}/{model_name}{FLAGS.scenario}.json')
            models[i] = model

    # Prepare Plots
    ## Plot specifications go here
    # ----------------------------
    fig, ax = plt.subplots(6, 7, figsize=(20,20), 
                           subplot_kw={
                               'xticks':[],  
                               'yticks':[]}
                            )

    # Start Plotting
    # ----------------------------
    for row, scene in enumerate( list(in_region.take(6)) ):
        img, tgt, _ = scene
        
        ax[row][0].title.set_text("Co-event Intensity (VH)")
        ax[row][0].imshow(img[0,:,:,0], cmap="gray") 


        ax[row][1].title.set_text("Ground Truth Label")
        ax[row][1].imshow(tgt[0], cmap=correct_cmap, interpolation='none')

        for col, model in enumerate(models):
            filtered_img = np.array(img[0,:,:,:])
            filtered_img = np.transpose(filtered_img, axes=(2,0,1)) # CHW

            if FLAGS.scenario == 1:
                filtered_img[:, :, :] = FILTERS[col](filtered_img[:, :, :])
                
            if FLAGS.scenario in [2,3]:
                filtered_img[0:4, :, :] = FILTERS[col](filtered_img[0:4, :, :])
            
            filtered_img = filtered_img[np.newaxis, ...]

            filtered_img = np.transpose(filtered_img, axes=(0, 2, 3, 1)) # Back to BHWC
            
            if (FLAGS.architecture == "unet"):
                logits = model(filtered_img) # Do we need to remove batch dimension?
                pred = tf.argmax(logits, axis=3) # BHWC
                fn_mask = np.ma.masked_where(pred==1, pred)
                fp_mask = np.ma.masked_where(tgt==1, pred)

                ax[row][2+col].title.set_text(f"UNet S{FLAGS.scenario}, {FILTER_NAMES[col]}")
                ax[row][2+col].imshow(pred[0], cmap=correct_cmap, interpolation='none')
                ax[row][2+col].imshow(np.ma.masked_array(tgt[0], fn_mask[0]), cmap=missing_cmap, interpolation='none')
                ax[row][2+col].imshow(fp_mask[0], cmap=wrong_cmap, interpolation='none')
            
            if(FLAGS.architecture == "xgb"):
                filtered_img = filtered_img[0,:,:,:]
                filtered_img = np.reshape(filtered_img, newshape=(512*512, -1))
                
                pred = model.model.predict(filtered_img)
                pred = np.array(pred)
                pred = np.reshape(pred, (512,512))
                pred = pred[np.newaxis, ...]

                fn_mask = np.ma.masked_where(pred==1, pred)
                fp_mask = np.ma.masked_where(tgt==1, pred)
                

                ax[row][2+col].title.set_text(f"XGBoost S{FLAGS.scenario}, {FILTER_NAMES[col]}")
                ax[row][2+col].imshow(pred[0], cmap=correct_cmap, interpolation='none')
                ax[row][2+col].imshow(np.ma.masked_array(tgt[0], fn_mask[0]), cmap=missing_cmap, interpolation='none')
                ax[row][2+col].imshow(fp_mask[0], cmap=wrong_cmap, interpolation='none')


            



    
    fig.savefig(f"/workspaces/Thesis/Results/Image Grids/{FLAGS.architecture}/{FLAGS.scenario}")

if __name__ == "__main__":
    app.run(main)




