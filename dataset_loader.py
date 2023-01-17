from collections import defaultdict
from dataclasses import dataclass, field
import os
from absl import app, flags
import numpy as np
from sklearn.model_selection import train_test_split

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

@dataclass
class Dataset:
    '''
    Holdout is Sri-Lanka.
    ? Hand labelled dataset means that the coherence was hand generated rather than using Alaska Satellite Facility's API.
    '''
    x_train: np.ndarray
    y_train: np.ndarray
    x_holdout: np.ndarray
    y_holdout: np.ndarray
    x_hand: np.ndarray
    y_hand: np.ndarray

    x_val: np.ndarray = field(init=False)
    y_val: np.ndarray = field(init=False)
    x_test: np.ndarray = field(init=False)
    y_test: np.ndarray = field(init=False)

    batches: dict = field(init=False)

    def __post_init__(self):
        # 70 : 20 : 10  train | test | val 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train, test_size=0.30)
        print(self.x_train.shape, self.y_train.shape)
        
        self.x_test, self.x_val, self.y_test, self.y_val,  = train_test_split(self.x_test, self.y_test, test_size=0.33)
    
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
    
    def load_batch(self):
        # Actually this will be different for each training method (Tensorflow vs XGBoost).
        # Probably better to just let them manipulate this Dataset class instead of doing it here.
        ...
        
def create_dataset(scenario:int=0) -> Dataset:
    '''
    Looks through indexed dataset folders to ensure that it creates a dataset where the same instances are available in every training scenario.
    Returns
    --  x_train :   A list of [filepaths] to TIF files to use for training
    --  y_train :   A list of filepaths to TIF files to use for labels
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

    files = index_dataset()
    suffixes = []
    hand_suffixes = []
    if scenario == 0:   # 2 data channels
        suffixes = [file_suffixes['s1_co']] 
    if scenario == 1:   # 4 data channels
        suffixes = [file_suffixes['s1_co'], file_suffixes['s1_pre']]
    if scenario == 2:   # 6 data channels.
        suffixes = [file_suffixes['s1_co'], file_suffixes['s1_pre'], file_suffixes['coh_co'], file_suffixes['coh_pre']]
        hand_suffixes = suffixes = [file_suffixes['s1_co'], file_suffixes['s1_pre'], file_suffixes['hand_co'], file_suffixes['hand_pre']]

    for k in files['coh_co'].keys():
        if files['coh_pre'][k] and files['s2_weak'][k] and files['s1_co'][k] and files['s1_pre'][k]:
            
            if k.split('_')[0]==holdout_region:
                x_holdout.append([[k+suf for suf in suffixes]])
                y_holdout.append(k+file_suffixes['s2_weak'])

            else:
                x_train.append([[k+suf for suf in suffixes]])
                y_train.append(k+file_suffixes['s2_weak'])

    for k in files['hand_co'].keys():
        if files['hand_pre'][k] and files['s2_weak'][k] and files['s1_co'][k] and files['s1_pre'][k]:
            
            if k.split('_')[0]==holdout_region:
                x_holdout.append([[k+suf for suf in hand_suffixes]])
                y_holdout.append(k+file_suffixes['s2_weak'])

            else:
                x_hand.append([[k+suf for suf in hand_suffixes]])
                y_hand.append(k+file_suffixes['s2_weak'])

    return Dataset( np.array(x_train), np.array(y_train), np.array(x_holdout), np.array(y_holdout), np.array(x_hand), np.array(y_hand) )

def index_dataset():
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

def main(x):
    dataset = create_dataset(FLAGS.scenario)

    print('Base dataset', dataset.x_train.shape + dataset.x_val.shape + dataset.x_test.shape)
    print('Hand dataset', dataset.x_hand.shape, dataset.y_hand.shape)
    print('Holdout dataset', dataset.x_holdout.shape, dataset.y_holdout.shape)

    print('\n Base train', dataset.x_train.shape, dataset.y_train.shape)
    print('Base val', dataset.x_val.shape, dataset.y_val.shape)
    print('Base test', dataset.x_test.shape, dataset.y_test.shape)

    dataset.generate_batches(4)

    for k in dataset.batches.keys():
        print( dataset.batches[k]['x'].shape, dataset.batches[k]['y'].shape)
        
    

if __name__ == "__main__":
    app.run(main)