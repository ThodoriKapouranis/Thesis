from collections import defaultdict
import os
from absl import app, flags
import numpy as np

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

def create_dataset(scenario:int=0):
    '''
    Looks through indexed dataset folders to ensure that it creates a dataset where the same instances are available in every training scenario.
    Returns
    --  x_train :   A list of [filepaths] to TIF files to use for training
    --  y_train :   A list of filepaths to TIF files to use for labels
    '''

    x_train = []
    y_train = []

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
    if scenario == 0:
        suffixes = [file_suffixes['s1_co']]
    if scenario == 1:
        suffixes = [file_suffixes['s1_co'], file_suffixes['s1_pre']]
    if scenario == 2:
        suffixes = [file_suffixes['s1_co'], file_suffixes['s1_pre'], file_suffixes['coh_co']]

    # use s1_co and hand_co to search other files

    for k in files['coh_co'].keys():
        if files['coh_pre'][k] and files['s2_weak'][k] and files['s1_co'][k] and files['s1_pre'][k]:
            x_train.append([[k+suf for suf in suffixes]])
            y_train.append(k+file_suffixes['s2_weak'])
    
    for k in files['hand_co'].keys():
        if files['s2_weak'][k] and files['s1_co'][k] and files['s1_pre'][k]:
            x_train.append([[k+suf for suf in suffixes]])
            y_train.append(k+file_suffixes['s2_weak'])

    return np.array(x_train), np.array(y_train)

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
    x_train, y_train = create_dataset(scenario=0)
    # print(x_train)
    print(x_train.shape, y_train.shape)
    print(x_train[0])

if __name__ == "__main__":
    app.run(main)