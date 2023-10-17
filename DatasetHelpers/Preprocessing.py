'''
Functions defined for the preprocessing pipeline are defined here

Looking to implement

-   Border noise corrections
-   Mono and Bitemporal Speckle filters (Lee, Refined Lee)

'''
from collections import defaultdict
from dataclasses import dataclass, field
import logging
import os
from typing import Tuple
from absl import app, flags
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import gridspec

import numpy as np
from sklearn.model_selection import train_test_split
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2 as cv

def lee_filter(image:np.ndarray, size:int = 5) -> np.ndarray:
    """Applies lee filter to image. It is applied per channel.

    https://www.imageeprocessing.com/2014/08/lee-filter.html
    https://www.kaggle.com/code/samuelsujith/lee-filter

    Args:
        image (np.array): Unfiltered image. Example size: (2,512,512)
        size (int, optional): Kernel size (N by N). Should be odd in order to have a 'center'. Defaults to 7.
    
    Returns:
        np.ndarray: Filtered image
    """
    EPSILON = 1e-9
    filtered = np.zeros(image.shape)
    
    # Apply filter to each channel wise
    for c in range(image.shape[0]):

        avg_kernel = np.ones((size, size), np.float32) / (size**2)
        
        patch_means = cv.filter2D(image[c], -1, avg_kernel)
        patch_means_sqr = cv.filter2D(image[c]**2, -1, avg_kernel)
        patch_var = patch_means_sqr - patch_means**2

        img_var = np.mean(image[c]**2) - np.mean(image[c])**2
        patch_weights = patch_var / (patch_var + img_var + EPSILON)
        filtered[c] = patch_means + patch_weights * (image[c] - patch_means)

    return filtered



def box_filter(image:np.ndarray, size:int=5) -> np.ndarray:
    avg_kernel = np.ones( shape=(size,size), dtype=np.float32) / (size**2)
    return cv.filter2D(image, -1, avg_kernel)

def PyRAT_rlf(image, *args, **kwargs):

    para = [
        {'var': 'win', 'value': 7, 'type': 'int', 'range': [3, 999], 'text': 'Window size'},
        {'var': 'looks', 'value': 1.0, 'type': 'float', 'range': [1.0, 99.0], 'text': '# of looks'},
        {'var': 'threshold', 'value': 0.5, 'type': 'float', 'range': [0.0, 9.0], 'text': 'threshold'},
        {'var': 'method', 'value': 'original', 'type': 'list', 'range': ['original', 'cov'], 'text': 'edge detector'}
    ]

    win = 7
    looks = 1.0
    threshold = 0.5
    method = 'original'
    filtered = np.zeros(image.shape, dtype=np.float32)
    
    for c in range(image.shape[0]):
        array = image[c, :, :] # <- Current Channel we are filtering on

        array[np.isnan(array)] = 0.0
        shape = array.shape
        if len(shape) == 3:
            array = np.abs(array)
            span = np.sum(array ** 2, axis=0)
            array = array[np.newaxis, ...]
        elif len(shape) == 4:
            span = np.abs(np.trace(array, axis1=0, axis2=1))
        else:
            array = np.abs(array)
            span = array ** 2
            array = array[np.newaxis, np.newaxis, ...]
        lshape = array.shape[0:2]

        # ---------------------------------------------
        # INIT & SPAN
        # ---------------------------------------------

        sig2 = 1.0 / looks
        sfak = 1.0 + sig2

        # nrx = array.shape[-1]
        #
        # lshape = array.shape[0:-2]
        # if len(lshape) == 2:
        # # span = np.abs(np.trace(array,axis1=0,axis2=1))
        #     span = np.abs(array[0, 0, ...] + array[1, 1, ...] + array[2, 2, ...])
        # else:
        #     logging.error("Data not in matrix form")

        # ---------------------------------------------
        # TURNING BOX
        # ---------------------------------------------

        cbox = np.zeros((9, win, win), dtype='float32')
        chbox = np.zeros((win, win), dtype='float32')
        chbox[0:win // 2 + 1, :] = 1
        cvbox = np.zeros((win, win), dtype='float32')
        for k in range(win):
            cvbox[k, 0:k + 1] = 1

        cbox[0, ...] = np.rot90(chbox, 3)
        cbox[1, ...] = np.rot90(cvbox, 1)
        cbox[2, ...] = np.rot90(chbox, 2)
        cbox[3, ...] = np.rot90(cvbox, 0)
        cbox[4, ...] = np.rot90(chbox, 1)
        cbox[5, ...] = np.rot90(cvbox, 3)
        cbox[6, ...] = np.rot90(chbox, 0)
        cbox[7, ...] = np.rot90(cvbox, 2)
        for k in range(win // 2 + 1):
            for l in range(win // 2 - k, win // 2 + k + 1):
                cbox[8, k:win - k, l] = 1

        for k in range(9):
            cbox[k, ...] /= np.sum(cbox[k, ...])

        ampf1 = np.empty((9,) + span.shape)
        ampf2 = np.empty((9,) + span.shape)
        for k in range(9):
            ampf1[k, ...] = sp.ndimage.correlate(span ** 2, cbox[k, ...])
            ampf2[k, ...] = sp.ndimage.correlate(span, cbox[k, ...]) ** 2

        # ---------------------------------------------
        # GRADIENT ESTIMATION
        # ---------------------------------------------
        np.seterr(divide='ignore', invalid='ignore')

        if method == 'original':
            xs = [+2, +2, 0, -2, -2, -2, 0, +2]
            ys = [0, +2, +2, +2, 0, -2, -2, -2]
            samp = sp.ndimage.uniform_filter(span, win // 2)
            grad = np.empty((8,) + span.shape)
            for k in range(8):
                grad[k, ...] = np.abs(np.roll(np.roll(samp, ys[k], axis=0), xs[k], axis=1) / samp - 1.0)
            magni = np.amax(grad, axis=0)
            direc = np.argmax(grad, axis=0)
            direc[magni < threshold] = 8
        elif method == 'cov':
            grad = np.empty((8,) + span.shape)
            for k in range(8):
                grad[k, ...] = np.abs((ampf1[k, ...] - ampf2[k, ...]) / ampf2[k, ...])
                direc = np.argmin(grad, axis=0)
        else:
            logging.error("Unknown method!")

        np.seterr(divide='warn', invalid='warn')
        # ---------------------------------------------
        # FILTERING
        # ---------------------------------------------
        out = np.empty_like(array)
        dbox = np.zeros((1, 1) + (win, win))
        for l in range(9):
            grad = ampf1[l, ...]
            mamp = ampf2[l, ...]
            dbox[0, 0, ...] = cbox[l, ...]

            vary = (grad - mamp).clip(1e-10)
            varx = ((vary - mamp * sig2) / sfak).clip(0)
            kfac = varx / vary
            if np.iscomplexobj(array):
                mamp = sp.ndimage.correlate(array.real, dbox) + 1j * sp.ndimage.convolve(array.imag,
                                                                                                        dbox)
            else:
                mamp = sp.ndimage.correlate(array, dbox)
            idx = np.argwhere(direc == l)
            out[:, :, idx[:, 0], idx[:, 1]] = (mamp + (array - mamp) * kfac)[:, :, idx[:, 0], idx[:, 1]]

        filtered[c] = out
    return filtered


def _test():
    from Dataset import create_dataset
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

    # Generate a whole bunch of lee filtered images
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

            # Transpose to get into HWC format. Psuedo color
            ax0 = np.transpose( np.array((original[:,:,0], original[:,:,1], original[:,:,0]/original[:,:,1])), (1,2,0))
            ax1 = np.transpose( np.array((filtered[:,:,0], filtered[:,:,1], filtered[:,:,0]/filtered[:,:,1])), (1,2,0))

            def normalize_channels(img):
                # Takes in HWC format
                # Scales image to 0...1
                for c in range(img.shape[2]):
                    diff_off_min = (img[:,:,c]-np.min(img[:,:,c]))
                    channel_range = (np.max(img[:,:,c])-np.min(img[:,:,c])) + 1e-3

                    img[:,:,c] = diff_off_min / channel_range
                
                return img
            
            ax0 = normalize_channels(ax0)
            ax1 = normalize_channels(ax1)

            axes[0].imshow(ax0)
            axes[1].imshow(ax1)

            fig.savefig(f'DatasetHelpers/pipeline-debugging/samples/lee-filter/{original_co_path.split("/")[-1][0:-3]}')

            plt.close()

    # Test Lee Filter functionality per-channel
    lee_test = np.zeros(shape=(6,60,60)) + np.random.normal(0, 0.1, size=(6,60,60))
    lee_test[0, 0:10, 0:10] += 1
    lee_test[1, 10:20, 10:20] += 1
    lee_test[2, 20:30, 20:30] += 1
    lee_test[3, 30:40, 30:40] += 1
    lee_test[4, 40:50, 40:50] += 1
    lee_test[5, 50:60, 50:60] += 1

    filtered = lee_filter(lee_test, size=7)
    
    fig, axes = plt.subplots(2, 6, figsize=(15,5))

    for i in range(6):
        axes[0,i].imshow(lee_test[i, :, :], interpolation='none')
        axes[1,i].imshow(filtered[i, :, :], interpolation='none')
    
    

    fig.savefig(f'DatasetHelpers/pipeline-debugging/filters-function/lee-test')
    plt.close()
    
def _test_refined():
    
    # Not working filter I tried to write heh..
    def refined_lee_filter(image:np.ndarray) -> np.ndarray:
        #TODO: Remember to convert DB -> Linear
        # https://github.com/adugnag/gee_s1_ard/blob/main/python-api/speckle_filter.py#L230
        # Polarimetric Radar Imaging: From Basics to Applications Page 150-152
        
        filtered = np.zeros(image.shape)

        for c in range(image.shape[0]):
            # Create a 3x3 Kernel to calculate the submeans of the image 
            subkernel_means = np.ones((3, 3), np.float32) / (3**2)
            
            # Get subpatch statistics
            patch_means = cv.filter2D(image[c], -1, subkernel_means)
            patch_means_sqr = cv.filter2D(image[c]**2, -1, subkernel_means)
            patch_var = patch_means_sqr - (patch_means**2)

            # Goes through the patch_means to grab the "center mean" of each 3x3 section
            main_kernel_sampler = [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]

            # Find a replacement for neighborhood to bands
            # Create 9 bands for each of the 9 3x3 sub-patches 
            # HARDCODING ALERT
            kernel_sampler_0 = np.zeros( (7,7), np.float32)
            kernel_sampler_1 = np.zeros( (7,7), np.float32)
            kernel_sampler_2 = np.zeros( (7,7), np.float32)
            kernel_sampler_3 = np.zeros( (7,7), np.float32)
            kernel_sampler_4 = np.zeros( (7,7), np.float32)
            kernel_sampler_5 = np.zeros( (7,7), np.float32)
            kernel_sampler_6 = np.zeros( (7,7), np.float32)
            kernel_sampler_7 = np.zeros( (7,7), np.float32)
            kernel_sampler_8 = np.zeros( (7,7), np.float32)

            kernel_sampler_0[1,1] = 1
            kernel_sampler_1[1,3] = 1
            kernel_sampler_2[1,5] = 1
            kernel_sampler_3[3,1] = 1
            kernel_sampler_4[3,3] = 1
            kernel_sampler_5[3,5] = 1
            kernel_sampler_6[5,1] = 1
            kernel_sampler_7[5,3] = 1
            kernel_sampler_8[5,5] = 1

            # mean patch sampler (each patch is a band)
            mean_bands = np.zeros(shape=(9, *image[c].shape), dtype=np.float32)
            mean_bands[0] = cv.filter2D(patch_means, -1, kernel_sampler_0)
            mean_bands[1] = cv.filter2D(patch_means, -1, kernel_sampler_1)
            mean_bands[2] = cv.filter2D(patch_means, -1, kernel_sampler_2)
            mean_bands[3] = cv.filter2D(patch_means, -1, kernel_sampler_3)
            mean_bands[4] = cv.filter2D(patch_means, -1, kernel_sampler_4)
            mean_bands[5] = cv.filter2D(patch_means, -1, kernel_sampler_5)
            mean_bands[6] = cv.filter2D(patch_means, -1, kernel_sampler_6)
            mean_bands[7] = cv.filter2D(patch_means, -1, kernel_sampler_7)
            mean_bands[8] = cv.filter2D(patch_means, -1, kernel_sampler_8)

            # variance patch sampler (each patch is a band)
            var_bands = np.zeros(shape=(9, *image[c].shape), dtype=np.float32)
            var_bands[0] = cv.filter2D(patch_var, -1, kernel_sampler_0)
            var_bands[1] = cv.filter2D(patch_var, -1, kernel_sampler_1)
            var_bands[2] = cv.filter2D(patch_var, -1, kernel_sampler_2)
            var_bands[3] = cv.filter2D(patch_var, -1, kernel_sampler_3)
            var_bands[4] = cv.filter2D(patch_var, -1, kernel_sampler_4)
            var_bands[5] = cv.filter2D(patch_var, -1, kernel_sampler_5)
            var_bands[6] = cv.filter2D(patch_var, -1, kernel_sampler_6)
            var_bands[7] = cv.filter2D(patch_var, -1, kernel_sampler_7)
            var_bands[8] = cv.filter2D(patch_var, -1, kernel_sampler_8)


            # Calculate all diagonals
            # This is the gradient values at each pixel
            gradients = np.zeros( shape=(4, *image[c].shape), dtype=np.float32)
            gradients[0] = np.abs(mean_bands[0] - mean_bands[8])
            gradients[1] = np.abs(mean_bands[1] - mean_bands[7])
            gradients[2] = np.abs(mean_bands[2] - mean_bands[6])
            gradients[3] = np.abs(mean_bands[5] - mean_bands[3])

            grad_max = np.max(gradients, axis=0)
            grad_max = np.stack( (grad_max, grad_max, grad_max, grad_max))
            grad_mask = grad_max != gradients # Keep only max gradients

            grad_mask = np.vstack( (grad_mask, grad_mask))

            ##TODO: Duplicate grad mask | gradient represents 2 directions. 
            # This means it will be 8 bands, so applying 8 masking bands to 8 direction bands
            # Should give us a single unique value for each centre pixel.

            ## Determine the 8 directions
            ## Telling the direction of the edge by taking 3 patch means in a 'row' and seeing the increase/decrease
            ## 'row' as in tic tac toe kinda row
            directions = np.zeros(shape=(8, *image[c].shape), dtype=np.float32)
            directions[0] = 1 * np.greater( (mean_bands[0] - mean_bands[4]), (mean_bands[4] - mean_bands[8]) )
            directions[1] = 2 * np.greater( (mean_bands[1] - mean_bands[4]), (mean_bands[4] - mean_bands[7]) )
            directions[2] = 3 * np.greater( (mean_bands[2] - mean_bands[4]), (mean_bands[4] - mean_bands[6]) )
            directions[3] = 4 * np.greater( (mean_bands[5] - mean_bands[4]), (mean_bands[4] - mean_bands[3]) )
            
            # The other directions are just reversed
            directions[4] = 5 * (directions[0]==0)
            directions[5] = 6 * (directions[1]==0)
            directions[6] = 7 * (directions[2]==0)
            directions[7] = 8 * (directions[3]==0)


            # Combine gradient mask with the directions
            # This will select the highest gradient and direction to use.
            directions = np.ma.masked_array(directions, mask=grad_mask, fill_value=0)
            directions = np.sum(directions, axis=0)

            # DE-BUG Visualize
            filtered[c] = directions
            # print(np.max(directions), np.min(directions))
        
            rect_w = np.array([
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
            ])

            diag_w = np.array([
                [1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
            ])

            # Dear Lord help me 
            dir_mean = np.ma.zeros(shape=(8, *image[c].shape), dtype=np.float32)
            dir_var = np.ma.zeros(shape=(8, *image[c].shape), dtype=np.float32)

            dir_mean[0] = np.ma.masked_array(cv.filter2D(image[c], -1, np.rot90(diag_w, 3)), mask=directions==1) 
            dir_var[0] = np.ma.masked_array(cv.filter2D(image[c]**2, -1, np.rot90(diag_w, 3)), mask=directions==1) - dir_mean[0]**2

            dir_mean[1] = np.ma.masked_array(cv.filter2D(image[c], -1, np.rot90(rect_w, 3)), mask=directions==2) 
            dir_var[1] = np.ma.masked_array(cv.filter2D(image[c]**2, -1, np.rot90(rect_w, 3)), mask=directions==2) - dir_mean[1]**2

            dir_mean[2] = np.ma.masked_array(cv.filter2D(image[c], -1, np.rot90(diag_w, 2)), mask=directions==3) 
            dir_var[2] = np.ma.masked_array(cv.filter2D(image[c]**2, -1, np.rot90(diag_w, 2)), mask=directions==3) - dir_mean[2]**2

            dir_mean[3] = np.ma.masked_array(cv.filter2D(image[c], -1, np.rot90(rect_w, 2)), mask=directions==4) 
            dir_var[3] = np.ma.masked_array(cv.filter2D(image[c]**2, -1, np.rot90(rect_w, 2)), mask=directions==4) - dir_mean[3]**2

            dir_mean[4] = np.ma.masked_array(cv.filter2D(image[c], -1, np.rot90(diag_w, 1)), mask=directions==5) 
            dir_var[4] = np.ma.masked_array(cv.filter2D(image[c]**2, -1, np.rot90(diag_w, 1)), mask=directions==5) - dir_mean[4]**2

            dir_mean[5] = np.ma.masked_array(cv.filter2D(image[c], -1, np.rot90(rect_w, 1)), mask=directions==6) 
            dir_var[5] = np.ma.masked_array(cv.filter2D(image[c]**2, -1, np.rot90(rect_w, 1)), mask=directions==6) - dir_mean[5]**2

            dir_mean[6] = np.ma.masked_array(cv.filter2D(image[c], -1, diag_w), mask=directions==7)
            dir_var[6] = np.ma.masked_array(cv.filter2D(image[c]**2, -1, diag_w), mask=directions==7) - dir_mean[6]**2

            dir_mean[7] = np.ma.masked_array(cv.filter2D(image[c], -1, rect_w), mask=directions==8)
            dir_var[7] = np.ma.masked_array(cv.filter2D(image[c]**2, -1, rect_w), mask=directions==8) - dir_mean[7]**2

            dir_mean = np.sum(dir_mean, axis=0)
            dir_var = np.sum(dir_var, axis=0)
            
            filtered[c] = dir_var

            # Noise variance in 8 bands (patches)
            noise_var_bands = np.divide(var_bands, mean_bands**2) 

            # Get the average noise variance using all patches (along axis=0)
            #TODO: No idea if this is what you're supposed to do. Can't make sense of the toArray().arraySort().arraySlice(0,0,5)
            noise_var = np.mean(noise_var_bands, axis=0)
            noise_var = 0.1**2

            x_var = (dir_var - dir_mean**2 * noise_var)/(1 + noise_var)

            # Weight
            b = x_var / dir_var 
            print(np.min(b), np.max(b))
            ## x =  ybar - b(y - ybar)
            x = dir_mean + b * (image[c] - dir_mean)

            filtered[c] = x_var
            
        return filtered 
   

    ## Adapted from https://github.com/birgander2/PyRAT/blob/master/pyrat/filter/Despeckle.py#L354C10-L354C10
    
    s = 10
    sample = np.zeros(shape=(6,64,64)) + np.random.normal(0, 0.1, size=(6,64,64))
    
    sample[0, 0:s*1, 0:s*1] += 1
    sample[1, s*1:s*2, s*1:s*2] += 1
    sample[2, s*2:s*3, s*2:s*3] += 1
    sample[3, s*3:s*4, s*3:s*4] += 1
    sample[4, s*4:s*5, s*4:s*5] += 1
    sample[5, s*5:s*6, s*5:s*6] += 1

    filtered = PyRAT_rlf(sample)
    
    fig, axes = plt.subplots(2, 6, figsize=(15,5))

    for i in range(6):
        axes[0,i].imshow(sample[i, :, :], interpolation='none')
        axes[1,i].imshow(filtered[i,:, :], interpolation='none')

    fig.savefig(f'DatasetHelpers/pipeline-debugging/filters-function/refined-lee-test')

def main(x):
    _test_refined()

if __name__ == "__main__":
    app.run(main)