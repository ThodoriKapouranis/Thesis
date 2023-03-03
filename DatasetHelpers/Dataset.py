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
    
    def generate_batches(self, batch_count:int) -> dict:
        '''
        Generates batches under the class attribute 'batches'
        It is a dictionary of batch idx that map to dictioanies with a 'x' and 'y' key.
        '''
        self.batches = {}
        it = len(self.x_train)/batch_count  # 1000/4 = 250
        batch_idx = [int(i*it) for i in range(batch_count)] # [0, 250, 500, 750]
        batch_idx.append(len(self.x_train)) # [0, 250, 500, 750, 1000]
        for i in range(batch_count):
            self.batches[i] = {}
            self.batches[i]['x'] = self.x_train[ batch_idx[i]:batch_idx[i+1] ]
            self.batches[i]['y'] = self.y_train[ batch_idx[i]:batch_idx[i+1] ]

        return self.batches        

def convert_to_tfds(ds:Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    '''
    Returns the created dataset as a tf.data.Dataset class.
    Returns:
        --  train_ds:     tf.data.Dataset
        --  val_ds  :     tf.data.Dataset
        --  test_ds (holdout) :     tf.data.Dataset
        // hand_ds : tf.data.Dataset
    '''
    # Samples will be converted to a list of string paths where the last string is the test label path
    train_samples = []
    test_samples = []
    val_samples = []

    for x, y in zip(ds.x_train, ds.y_train): train_samples.append((*x, *y))
    for x, y in zip(ds.x_holdout, ds.y_holdout): test_samples.append((*x, *y)) 
    for x, y in zip(ds.x_val, ds.y_val): val_samples.append((*x, *y))
    
    train_samples, test_samples, val_samples = np.asarray(train_samples), np.asarray(test_samples), np.asarray(val_samples)

    train_ds = tf.data.Dataset.from_tensor_slices(train_samples)
    train_ds = train_ds.map(tf_read_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.map(load_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(test_samples)
    test_ds = test_ds.map(tf_read_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(load_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(val_samples)
    val_ds = val_ds.map(tf_read_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(load_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return train_ds, val_ds, test_ds
    # return train_ds, None, None

def read_sample(data_path:str) -> tuple:
    # Used by tf_read_sample to show tensorflow how to load our data in its own automatic batching process.
    #! Hardcode this for now
    #! Please please please figure out how to change this later. yucky
    CLASS_W = {0: 0.6212519560516805, 1: 2.5618224079902174} 
    path = data_path.numpy() # 0:-1 --> training paths
    img = list()
    tgt = list()

    for train_path in path[0:-1]:
        train_path = train_path.decode('utf-8')
        with rasterio.open(train_path) as src:
            tmp_img = src.read()
            img.append(tmp_img)
    
    img = np.squeeze(np.asarray(img))
    img = np.transpose(img, axes=(1,2,0))
    img = np.expand_dims(img, axis=0)

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
    
    # ## MASKING
    nans = np.isnan(img[0,:,:,:]).any(axis=-1)
    tgt_masked = np.ma.masked_array(tgt, mask=nans)
    
    # ## NAN IMPUTATION
    if np.count_nonzero(nans) > 0:
        img = np.nan_to_num(img, nan=0.0)
    
    ## DEBUG MASKING
    # if np.count_nonzero(nans) > 0:
    #     # print(nans.shape, np.count_nonzero(nans))

    #     fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)

    #     ax1.imshow(nans, cmap='Greys')
    #     ax2.imshow(img[0,:,:,-1], cmap='Greys')
    #     ax3.imshow(tgt.reshape((512,512)), cmap='Greys')
    #     ax4.imshow(tgt_masked.reshape((512,512)), cmap='Greys')

    #     fig1.savefig(f"Results/Debug/{tgt_path.split('/')[-1][:-3]}_masks.png")

    # plt.close()

    tgt_masked = np.transpose(tgt_masked, axes=(1,2,0))
    tgt_masked = np.expand_dims(tgt_masked, axis=0)

    ## Add the weighting
    weights = np.ones(tgt_masked.shape, dtype=np.float32)
    for k,v in CLASS_W.items():
        weights[ tgt_masked == k] = v

    return (img, tgt_masked, weights)

@tf.function
def tf_read_sample(data_path:str) -> dict:
    [img, tgt, weight] = tf.py_function( read_sample, [data_path], [tf.float32, tf.float32, tf.float32])
    img.set_shape((1, 512, 512, 2))
    tgt.set_shape((1, 512, 512, 1))
    weight.set_shape((1, 512, 512, 1))
    return {'image': img, 'target': tgt, 'weight': weight}

@tf.function
def load_sample(sample: dict) -> tuple:
  # convert to tf image
  image = tf.image.resize(sample['image'], (512, 512))
  target = tf.image.resize(sample['target'], (512, 512))
  weight = tf.image.resize(sample['weight'], (512, 512))

  # cast to proper data types
  image = tf.cast(image, tf.float32)
  target = tf.cast(target, tf.float32)
  weight = tf.cast(weight, tf.float32)
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
        'hand_co' : '_co_event_coh_HandLabeled.tif',
        'hand_pre': '_pre_event_coh_HandLabeled.tif',
        'coh_co': '_co_event_coh.tif',
        'coh_pre': '_pre_event_coh.tif',
        's2_weak': '_S2IndexLabelWeak.tif',
    }

    file_dir = {
        's1_co' :FLAGS.s1_co,
        's1_pre' : FLAGS.s1_pre,
        'hand_co' : FLAGS.hand_co,
        'hand_pre': FLAGS.hand_pre,
        'coh_co': FLAGS.coh_co,
        'coh_pre': FLAGS.coh_pre,
        's2_weak': FLAGS.s2_weak,
    }

    files = index_dataset(FLAGS)
    suffixes = []
    hand_suffixes = []
    label_dir, label_suffix = file_dir['s2_weak'], file_suffixes['s2_weak']

    if FLAGS.scenario == 1:   # 2 data channels
        usable_data = ['s1_co']
        suffixes = [file_suffixes[scene_type] for scene_type in usable_data] 
        dirs = [file_dir[scene_type] for scene_type in usable_data] 
    
    if FLAGS.scenario == 2:   # 4 data channels
        usable_data = ['s1_co', 's1_pre']
        suffixes = [file_suffixes[scene_type] for scene_type in usable_data]
        dirs = [file_dir[scene_type] for scene_type in usable_data] 
    
    if FLAGS.scenario == 3:   # 6 data channels.
        usable_data = ['s1_co', 's1_pre', 'coh_co', 'coh_pre']
        suffixes = [file_suffixes[scene_type] for scene_type in usable_data]
        dirs = [file_dir[scene_type] for scene_type in usable_data] 
        hand_suffixes = [file_suffixes[scene_type] for scene_type in usable_data]

    for k in files['coh_co'].keys():
        # Ensure existence in each indexed directory
        if files['coh_pre'][k] and files['s2_weak'][k] and files['s1_co'][k] and files['s1_pre'][k]:
            
            if k.split('_')[0]==holdout_region:
                x_holdout.append([ dir + '/' + k + suf for dir, suf in zip(dirs, suffixes)])
                y_holdout.append([label_dir + '/' + k + label_suffix])

            else:
                x_train.append([ dir + '/' + k + suf for dir, suf in zip(dirs, suffixes)])
                y_train.append([label_dir + '/' + k + label_suffix])

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
        'hand_co': defaultdict(lambda: False),
        'hand_pre': defaultdict(lambda: False),
        's2_weak': defaultdict(lambda: False),
        'coh_co': defaultdict(lambda: False),
        'coh_pre':defaultdict(lambda: False)
    }

    is_tif = lambda x: True if x[-4:]==".tif" else False    
    
    # Creates a dictionary of every file that exists in the dataset to use for quick indexing
    for dict, folder in zip(['s1_co', 's1_pre', 'hand_co', 'hand_pre', 's2_weak', 'coh_co', 'coh_pre'], [FLAGS.s1_co, FLAGS.s1_pre, FLAGS.hand_co, FLAGS.hand_pre, FLAGS.s2_weak, FLAGS.coh_co, FLAGS.coh_pre]):
        for file in os.listdir(folder):
            if not is_tif(file):
                continue
            else:
                file = file.split('_')
                country, num = file[0], file[1]
                files[dict][country + '_' + num] = True
    
    return files

def _test():
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

    print('\nBase dataset', dataset.x_train.shape + dataset.x_val.shape + dataset.x_test.shape)
    print('Hand dataset', dataset.x_hand.shape, dataset.y_hand.shape)
    print('Holdout dataset', dataset.x_holdout.shape, dataset.y_holdout.shape)

    print('\n Base train', dataset.x_train.shape, dataset.y_train.shape)
    print('Base val', dataset.x_val.shape, dataset.y_val.shape)
    print('Base test', dataset.x_test.shape, dataset.y_test.shape)

    batches = dataset.generate_batches(4)
    print(f'\nCut into 4 batches with keys: {batches.keys()}')

    train_ds, _, _ = convert_to_tfds(dataset)

    for img, tgt, _ in train_ds.take(1):
        print(img.shape, tgt.shape)
        fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.imshow(img[0,:,:,0])
        ax2.imshow(img[0,:,:,1])
        fig1.savefig('Results/test.png')


def main(x):
    _test()
        
if __name__ == "__main__":
    app.run(main)