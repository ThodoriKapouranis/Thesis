from collections import defaultdict
from dataclasses import dataclass, field
import os
from typing import Tuple
from absl import app, flags
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2 as cv
import sys

sys.path.append(os.path.abspath("/workspaces/Thesis/DatasetHelpers/"))
from Preprocessing import lee_filter

label_remapping = {
    -1: 0,
    0: 0,
    1: 1,
}

@dataclass
class Dataset:
    '''
    Holdout is Sri-Lanka.
    ? Hand labelled dataset means that the coherence was hand generated rather than using Alaska Satellite Facility's API.
    '''
    scenario: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_holdout: np.ndarray   # Sri Lanka test set
    y_holdout: np.ndarray   # Sri Lanka test set
    x_hand: np.ndarray      # Hand labelled test set
    y_hand: np.ndarray      # Hand labelled test set

    x_val: np.ndarray = field(init=False) # Should be taken from x_train post_init
    y_val: np.ndarray = field(init=False) # Should be taken from y_train post_init
    # x_test: np.ndarray = field(init=False)
    # y_test: np.ndarray = field(init=False)

    channels: int = field(init=False)
    batches: dict = field(init=False)

    def __post_init__(self):
        # 70 : 20 : 10  train | test | val 
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.30)
        # self.x_test, self.x_val, self.y_test, self.y_val,  = train_test_split(self.x_test, self.y_test, test_size=0.33)
        
        if self.scenario == 0: self.channels = 2
        elif self.scenario == 1: self.channels = 4
        elif self.scenario == 2: self.channels = 6
        else: self.channels = None
    
    def generate_batches(self, batch_count:int, which_ds:str="train") -> dict:
        """Writes batch information under the self.batches attribute of this class.

        Batch information is stored as a dictionary that points to a list of data and label image paths under keys 'x' and 'y' 

        Args:
            batch_count (int): Number of batches to split training into
            which_ds (str): Which dataset split to use. "train" "hand" "holdout"

        Returns:
            dict: self.batches 
        """
        self.batches = {}
        ds_x = self.x_train
        ds_y = self.y_train

        if which_ds == "hand":
            ds_x = self.x_hand
            ds_y = self.y_hand
        elif which_ds == "holdout":
            ds_x = self.x_holdout
            ds_y = self.y_holdout
        
        it = len(ds_x)/batch_count  # 1000/4 = 250
        batch_idx = [int(i*it) for i in range(batch_count)] # [0, 250, 500, 750]
        batch_idx.append(len(ds_x)) # [0, 250, 500, 750, 1000]
        for i in range(batch_count):
            self.batches[i] = {}
            self.batches[i]['x'] = ds_x[ batch_idx[i]:batch_idx[i+1] ]
            self.batches[i]['y'] = ds_y[ batch_idx[i]:batch_idx[i+1] ]

        return self.batches        

def convert_to_tfds(ds:Dataset, channel_size:int, filter:any, format:str='HWC') -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    '''
    Returns the created datasets as multiple tf.data.Dataset classes.
    Returns:
        --  train_ds:     tf.data.Dataset
        --  val_ds  :     tf.data.Dataset
        --  test_ds (holdout) :     tf.data.Dataset
        --  hand_ds : tf.data.Dataset
    '''
    # Samples will be converted to a list of string paths where the last string is the test label path
    train_samples = []
    val_samples = []
    test_samples = []
    hand_samples = []
    
    tf_read_sample = construct_read_sample_function(channel_size, filter=filter, format=format, )

    for x, y in zip(ds.x_train, ds.y_train): train_samples.append((*x, *y))
    for x, y in zip(ds.x_val, ds.y_val): val_samples.append((*x, *y))
    for x, y in zip(ds.x_holdout, ds.y_holdout): test_samples.append((*x, *y)) 
    for x, y in zip(ds.x_hand, ds.y_hand): hand_samples.append((*x, *y))
    
    train_samples, val_samples, test_samples, hand_samples = np.asarray(train_samples), np.asarray(val_samples), np.asarray(test_samples), np.asarray(hand_samples)

    train_ds = tf.data.Dataset.from_tensor_slices(train_samples)
    train_ds = train_ds.map(tf_read_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.map(load_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(val_samples)
    val_ds = val_ds.map(tf_read_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(load_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(test_samples)
    test_ds = test_ds.map(tf_read_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(load_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    hand_ds = tf.data.Dataset.from_tensor_slices(hand_samples)
    hand_ds = hand_ds.map(tf_read_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    hand_ds = hand_ds.map(load_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds, test_ds, hand_ds

def construct_read_sample_function(channel_size:int, filter, format:str = "HWC"):
    '''
    This function takes in options to adjust the dataset reading functions to accomodate different datasets.

    This is because the read_sample, etc, functions need to have an exactly specific argument list to work.
    We wrap the definitions of this function with another function to modify their behavior.
    
    @parmams:
        - channel_size = Channel size of the dataset (3 for rgb)
        - format : image dimension order. "HWC" or "CHW"
    '''
    
    def apply_transpose(x:np.float32):
        # Assume x is read directly from rasterio.open. Which means it would be in CHW format
        if format == "CHW":
            return x 
        elif format == "HWC":
            return np.transpose(x, axes=(0,2,3,1))
    
    def read_sample(data_path:str) -> tuple:
        # Used by tf_read_sample to show tensorflow how to load our data in its own automatic batching process.
        #! Hardcode this for now
        #! Please please please figure out how to change this later. yucky
        CLASS_W = {0: 0.6212519560516805, 1: 2.5618224079902174} 
        path = data_path.numpy() # 0:-1 --> training paths
        img = []
        tgt = list()

        for train_path in path[0:-1]:
            # Train paths include all images paths relating to this scene
            train_path = train_path.decode('utf-8')
            
            with rasterio.open(train_path) as src:
                tmp_img = src.read()
                
                # First image/"channel" to be appended to the list
                if img == []: 
                    img.append(tmp_img)
                    img = np.asarray(img) # --> (1, 2, 512, 512)
                
                # Subsequent channels will be np.appended to perserve shape
                else:
                    tmp_img = np.expand_dims(tmp_img, axis=0)
                    img = np.append(img, tmp_img, axis=1) # --> (1, 2+, 512, 512)


        tgt_path = path[-1].decode('utf-8')
        with rasterio.open(tgt_path) as src:
            tgt = src.read()
            
            for old_val, new_val in label_remapping.items():
                tgt[tgt == old_val] = new_val

        ## DEBUG TRAINING DATA
        # fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        # ax1.imshow(img[0,:,:,0], cmap='Greys')
        # ax2.imshow(img[0,:,:,1], cmap='Greys')
        # fig1.savefig(f"Results/Debug/{tgt_path.split('/')[-1][:-3]}_training.png")
        
        ## PREPROCESSING PIPLINE

        ##  ## MASKING
        # Get along channels
        nans = np.isnan(img[0,:,:,:]).any(axis=0) 

        ## ## NAN IMPUTATION for input
        # Is zero a good imputation value?
        if np.count_nonzero(nans) > 0:
            img = np.nan_to_num(img, nan=0.0)



        #### APPLY SPECKLE FILTER
        if img.shape[1] == 2:
            img[0, :, :, :] = filter(img[0, :, :, :])
            
        if img.shape[1] > 2:
            img[0, 0:4, :, :] = filter(img[0, 0:4, :, :])


            ##  ## RADIOMETRIC TERRAIN NORMALIZATION

        tgt_masked = np.ma.masked_array(tgt, mask=nans)
        

        # Apply appropriate transpose to get into correct final training format
        # Everything initially is BCWH
        img = apply_transpose(img) 


        
        ###### DEBUG MASKING
        # if np.count_nonzero(nans) > 0:
        #     # print(nans.shape, np.count_nonzero(nans))

        #     fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)

        #     ax1.imshow(nans, cmap='Greys')
        #     ax2.imshow(img[0,:,:,-1], cmap='Greys')
        #     ax3.imshow(tgt.reshape((512,512)), cmap='Greys')
        #     ax4.imshow(tgt_masked.reshape((512,512)), cmap='Greys')

        #     fig1.savefig(f"Results/Debug/{tgt_path.split('/')[-1][:-3]}_masks.png")

        # plt.close()

        tgt_masked = tgt_masked[0,:,:] # Remove channels

        ## Add the weighting
        weights = np.ones(tgt_masked.shape, dtype=np.float32)
        for k,v in CLASS_W.items():
            weights[ tgt_masked == k] = v

        # Remove batch
        ## Commenting this out so that legacy models work. What was the point of this?
        # img = img[0,:,:,:]

        return (img, tgt_masked, weights)

    @tf.function
    def tf_read_sample(data_path:str) -> dict:
        [img, tgt, weight] = tf.py_function( read_sample, [data_path], [tf.float32, tf.float32, tf.float32])
        # Adding None for batch size

        if format == "HWC":
            img.set_shape((None, 512, 512, channel_size))
            tgt.set_shape((512, 512))
            weight.set_shape((512, 512))
        
        elif format == "CHW":
            img.set_shape((None, channel_size, 512, 512))
            tgt.set_shape((512, 512))
            weight.set_shape((512, 512))

        return {'image': img, 'target': tgt, 'weight': weight}
    
    return tf_read_sample

    # @tf.function
    # def tf_read_sample(data_path:str) -> dict:
    #     [img, tgt, weight] = tf.py_function( read_sample, [data_path], [tf.float32, tf.float32, tf.float32])
    #     # todo: These shapes need to be changed given scenario flags
    #     img.set_shape((1, 512, 512, 2))
    #     tgt.set_shape((1, 512, 512, 1))
    #     weight.set_shape((1, 512, 512, 1))
    #     return {'image': img, 'target': tgt, 'weight': weight}

@tf.function
def load_sample(sample: dict) -> tuple:
  # convert to tf image
#   image = tf.image.resize(sample['image'], (512, 512))
#   target = tf.image.resize(sample['target'], (512, 512))
#   weight = tf.image.resize(sample['weight'], (512, 512))

  # cast to proper data types
  image = tf.cast(sample['image'], tf.float32)
  target = tf.cast(sample['target'], tf.float32) # Get rid of channel dimension
  weight = tf.cast(sample['weight'], tf.float32)
  return image, target, weight


def create_dataset(FLAGS:flags.FLAGS) -> Dataset:
    '''
    Looks through dataset folders to ensure that it creates a dataset where the same scene instances are available in ALL training scenarios.
    
    Returns
        -- Dataset: DatasetHelpers.Dataset

    In order to be usable with tensorflow NN models, Dataset.convert_to_tfds must be called.
    '''

    holdout_region = "Sri-Lanka"
    x_train = []
    y_train = []
    x_holdout = []
    y_holdout = []
    x_hand = []
    y_hand = []

    file_suffixes = {
        's1_co' : '_S1Weak.tif',        # <-- Why is this weak? I thought weak was supposed to mean thresholding on some basis to generate labels
        's1_pre' : '_pre_event_grd.tif',
        'coh_co': '_co_event_coh.tif',
        'coh_pre': '_pre_event_coh.tif',
        's2_weak': '_S2IndexLabelWeak.tif',

        'hand_coh_co': '_co_event_coh_HandLabeled.tif',
        'hand_coh_pre': '_pre_event_coh_HandLabeled.tif',
        'hand_s1_co': '_S1Hand.tif',
        'hand_s1_pre': '_pre_event_grd_HandLabeled.tif',
        'hand_labels': '_LabelHand.tif',
    }

    file_dir = {
        's1_co' :FLAGS.s1_co,
        's1_pre' : FLAGS.s1_pre,
        'coh_co': FLAGS.coh_co,
        'coh_pre': FLAGS.coh_pre,
        's2_weak': FLAGS.s2_weak,
        'hand_coh_co': FLAGS.hand_coh_co,
        'hand_coh_pre': FLAGS.hand_coh_pre,
        'hand_s1_co': FLAGS.hand_s1_co,
        'hand_s1_pre': FLAGS.hand_s1_pre,
        'hand_labels': FLAGS.hand_labels,
    }

    files = index_dataset(FLAGS)
    suffixes = []
    hand_suffixes = []
    label_dir, label_suffix = file_dir['s2_weak'], file_suffixes['s2_weak']

    if FLAGS.scenario == 1:   # 2 data channels 
        # Define the directories and name_suffixes to be used in creating the dataset
        # These details will be used to automatically generate all the filepaths of valid files
        usable_data = ['s1_co']
        suffixes = [file_suffixes[scene_type] for scene_type in usable_data] 
        dirs = [file_dir[scene_type] for scene_type in usable_data] 

        hand_suffixes = [file_suffixes['hand_' + scene_type] for scene_type in usable_data]
        hand_dirs = [file_dir['hand_' + scene_type] for scene_type in usable_data]

    if FLAGS.scenario == 2:   # 4 data channels
        usable_data = ['s1_co', 's1_pre']
        suffixes = [file_suffixes[scene_type] for scene_type in usable_data]
        dirs = [file_dir[scene_type] for scene_type in usable_data] 

        hand_suffixes = [file_suffixes['hand_' + scene_type] for scene_type in usable_data]
        hand_dirs = [file_dir['hand_' + scene_type] for scene_type in usable_data]
    
    if FLAGS.scenario == 3:   # 6 data channels.
        usable_data = ['s1_co', 's1_pre', 'coh_co', 'coh_pre']
        suffixes = [file_suffixes[scene_type] for scene_type in usable_data]
        dirs = [file_dir[scene_type] for scene_type in usable_data] 
        
        hand_suffixes = [file_suffixes['hand_' + scene_type] for scene_type in usable_data]
        hand_dirs = [file_dir['hand_' + scene_type] for scene_type in usable_data]

    # Main + holdout dataset
    for k in files['coh_co'].keys():
        # Ensure existence in each indexed directory
        if files['coh_pre'][k] and files['s2_weak'][k] and files['s1_co'][k] and files['s1_pre'][k]:
            
            # Create all filepaths from the previous information extracted from the given scenario
            if k.split('_')[0]==holdout_region:
                x_holdout.append([ dir + '/' + k + suf for dir, suf in zip(dirs, suffixes)])
                y_holdout.append([label_dir + '/' + k + label_suffix])

            else:
                x_train.append([ dir + '/' + k + suf for dir, suf in zip(dirs, suffixes)])
                y_train.append([label_dir + '/' + k + label_suffix])

    # Hand labelled
    for k in files['hand_coh_co'].keys():
        # Ensure valid existence in each indexed hand-labelled directory
        if files['hand_coh_pre'][k] and files['hand_labels'][k] and files['hand_s1_co'][k] and files['hand_s1_pre'][k]:
            # Create all filepaths from the previous information extracted from the given scenario
            x_hand.append([ dir + '/' + k + suf for dir, suf in zip(hand_dirs, hand_suffixes)])
            y_hand.append([file_dir['hand_labels'] + '/' + k + file_suffixes['hand_labels']])
    
    return Dataset( 
        FLAGS.scenario, 
        np.array(x_train), 
        np.array(y_train), 
        np.array(x_holdout), 
        np.array(y_holdout), 
        np.array(x_hand), 
        np.array(y_hand) 
    )

def index_dataset(FLAGS:flags.FLAGS):
    '''
    Returns a dictionary of all the TIF files in the folder, to quickly check for existence.
    Used by create_dataset() to only include training files who are available in every folder.
    Filenames have the gather method suffix removed.
    Bolivia_18962_co_event_coh.tif => Bolivia_16982
    '''
    files = { 
        's1_co': defaultdict(lambda: False),
        's1_pre': defaultdict(lambda: False),
        's2_weak': defaultdict(lambda: False),
        'coh_co': defaultdict(lambda: False),
        'coh_pre':defaultdict(lambda: False),
        'hand_coh_co':  defaultdict(lambda: False),
        'hand_coh_pre': defaultdict(lambda: False),
        'hand_s1_co':   defaultdict(lambda: False),
        'hand_s1_pre':  defaultdict(lambda: False),
        'hand_labels':  defaultdict(lambda: False),
    }   

    file_dir = {
        's1_co' :FLAGS.s1_co,
        's1_pre' : FLAGS.s1_pre,
        'coh_co': FLAGS.coh_co,
        'coh_pre': FLAGS.coh_pre,
        's2_weak': FLAGS.s2_weak,
        'hand_coh_co': FLAGS.hand_coh_co,
        'hand_coh_pre': FLAGS.hand_coh_pre,
        'hand_s1_co': FLAGS.hand_s1_co,
        'hand_s1_pre': FLAGS.hand_s1_pre,
        'hand_labels': FLAGS.hand_labels,
    }

    is_tif = lambda x: True if x[-4:]==".tif" else False    
    
    # Creates a dictionary of every folder to use for quick indexing / search for file existence
    for key, folder in file_dir.items():
        for file in os.listdir(folder):
            if not is_tif(file):
                continue
            else:
                file = file.split('_')
                country, num = file[0], file[1]
                files[key][country + '_' + num] = True

    return files

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

    dataset = create_dataset(FLAGS)

    print('\nBase train', dataset.x_train.shape, dataset.y_train.shape)
    print('Base val', dataset.x_val.shape, dataset.y_val.shape)
    print('Base test', dataset.x_holdout.shape, dataset.y_holdout.shape)

    print('Hand labelled', dataset.x_hand.shape, dataset.y_hand.shape)

    batches = dataset.generate_batches(4)
    print(f'\nCut into 4 batches with keys: {batches.keys()}')

    train_ds, _, _ = convert_to_tfds(dataset, 1)

    # for img, tgt, _ in train_ds.take(1):
    #     print(img.shape, tgt.shape)
    #     fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    #     ax1.imshow(img[0,:,:,0])
    #     ax2.imshow(img[0,:,:,1])
    #     fig1.savefig('Results/test.png')


def main(x):
    _test()
        
if __name__ == "__main__":
    app.run(main)