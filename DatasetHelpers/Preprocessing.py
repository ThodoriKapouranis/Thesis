'''
Functions defined for the preprocessing pipeline are defined here

Looking to implement

-   Border noise corrections
-   Mono and Bitemporal Speckle filters (Lee, Refined Lee)

'''
from collections import defaultdict
from dataclasses import dataclass, field
import os
from typing import Tuple
from absl import app, flags
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Dataset import create_dataset
import cv2 as cv

def lee_filter(image:np.ndarray, size:int = 7) -> np.ndarray:
    """Applies lee filter to image

    https://www.imageeprocessing.com/2014/08/lee-filter.html
    https://www.kaggle.com/code/samuelsujith/lee-filter

    Args:
        image (np.array): Unfiltered image. Example size: (2,512,512)
        size (int, optional): Kernel size (N by N). Should be odd in order to have a 'center'. Defaults to 7.
    
    Returns:
        np.ndarray: Filtered image
    """

    avg_kernel = np.ones((size, size), np.float32) / (size**2)
    
    patch_means = cv.filter2D(image, -1, avg_kernel)
    patch_means_sqr = cv.filter2D(image**2, -1, avg_kernel)
    patch_var = patch_means_sqr - patch_means**2

    img_var = np.mean(image**2) - np.mean(image)**2
    patch_weights = patch_var / (patch_var + img_var)

    filtered = patch_means + patch_weights * (image - patch_means)

    return filtered

# def lee_filter(img, size:int=7):
#     img_mean = uniform_filter(img, (size, size))
#     img_sqr_mean = uniform_filter(img**2, (size, size))
#     img_variance = img_sqr_mean - img_mean**2

#     overall_variance = variance(img)

#     img_weights = img_variance / (img_variance + overall_variance)
#     img_output = img_mean + img_weights * (img - img_mean)
#     return img_output


def debug_mean_filter(image:np.ndarray, N:int=10) -> np.ndarray:
    avg_kernel = np.ones( shape=(N,N), dtype=np.float32) / (N**2)

    return cv.filter2D(image, -1, avg_kernel)
def _test():
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("debug", False, "Set logging level to debug")
    flags.DEFINE_integer("scenario", 1, "Training data scenario. \n\t 1: Only co_event \n\t 2: coevent & preevent \n\t 3: coevent & preevent & coherence")
    flags.DEFINE_string("model", "xgboost", "'xgboost', 'unet', 'a-unet")
    flags.DEFINE_string('s1_co', '/workspaces/Thesis/10m_data/s1_co_event_grd', 'filepath of Sentinel-1 coevent data')
    flags.DEFINE_string('s1_pre', '/workspaces/Thesis/10m_data/s1_pre_event_grd', 'filepath of Sentinel-1 prevent data')
    flags.DEFINE_string('s2_weak', '/workspaces/Thesis/10m_data/s2_labels', 'filepath of S2-weak labelled data')
    flags.DEFINE_string('coh_co', '/workspaces/Thesis/10m_data/coherence/co_event', 'filepath of coherence coevent data')
    flags.DEFINE_string('coh_pre', '/workspaces/Thesis/10m_data/coherence/pre_event', 'filepath of coherence prevent data')

    # Hand labelled directories
    flags.DEFINE_string('hand_coh_co', '/workspaces/Thesis/10m_hand/coherence_10m/hand_labeled/co_event', '(h) filepath of coevent data')
    flags.DEFINE_string('hand_coh_pre', '/workspaces/Thesis/10m_hand/coherence_10m/hand_labeled/pre_event', '(h) filepath of preevent data')
    flags.DEFINE_string('hand_s1_co', '/workspaces/Thesis/10m_hand/HandLabeled/S1Hand', '(h) filepath of Sentinel-1 coevent data')
    flags.DEFINE_string('hand_s1_pre', '/workspaces/Thesis/10m_hand/S1_Pre_Event_GRD_Hand_Labeled', '(h) filepath of Sentinel-1 prevent data')
    flags.DEFINE_string('hand_labels', '/workspaces/Thesis/10m_hand/HandLabeled/LabelHand', 'filepath of hand labelled data')

    ds = create_dataset(FLAGS)

    # Generate a whole bunch of 
    random_index = np.random.randint( low=0, high=ds.x_train.shape[0]-1, size=(30) )

    for i in random_index:
        original_co_path = ds.x_train[i, 0]

        with rasterio.open(original_co_path) as src:
            original = src.read()

            nans = np.isnan(original[:,:,:]).any(axis=0) 

            img = original
            ## ## NAN IMPUTATION for input
            # Is zero a good imputation value?
            if np.count_nonzero(nans) > 0:
                img = np.nan_to_num(original, nan=0.0)

            
            # Apply pipeline
            filtered = lee_filter(img, size=9)

            fig, axes = plt.subplots(1, 2)

            # original = cv.normalize(original, None, alpha=0, beta=255, norm_type = cv.NORM_MINMAX)
            # filtered = cv.normalize(filtered, None, alpha=0, beta=255, norm_type = cv.NORM_MINMAX)

            # original = np.append(original, np.expand_dims(original[0]/original[1], 0), 0)
            # filtered = np.append(filtered, np.expand_dims(filtered[0]/filtered[1], 0), 0)
            
            # Transpose to get into HWC format

            original = np.transpose(original, (1,2,0))
            filtered = np.transpose(filtered, (1,2,0))


            ax0 = original[:,:,0]
            ax1 = filtered[:,:,0]

            axes[0].imshow(ax0)
            axes[1].imshow(ax1)

            fig.savefig(f'DatasetHelpers/pipeline-debugging/{original_co_path.split("/")[-1][0:-3]}')

            plt.close()


def main(x):
    _test()

if __name__ == "__main__":
    app.run(main)