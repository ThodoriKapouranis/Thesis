import xgboost as xgb
from collections import defaultdict
from dataclasses import dataclass, field
import os
from absl import app, flags
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

@dataclass
class Batched_XGBoost:
    model: any = field(init=False)
    
    def train_in_batches(self, batches:dict):
        '''
        This function trains xgboost models in batches in order to fit in the GPU's memory.
        The files are only loaded in memory once they are needed for training.
        Arguments:
        --  x_train :   A list of filepaths to TIF files to use for training
        --  y_train :   A list of filepaths to TIF files to use for labels

        Returns:
        --  model   :   Trained XGBoost model using the xgboost library
        --  hist    :   training info?
        '''
        full_model = None

        for batch_idx in batches.keys():
            
            batch_model = xgb.XGBClassifier(use_label_encoder=False, tree_method='gpu_hist', xgb_model=full_model)
            x, y = self.__load_data(batches[batch_idx])
            batch_model.fit(x, y, xgb_model=None)
            full_model = batch_model
        
        self.model =  full_model

    
    def __load_data(self, batch:dict):
        '''
        Takes a dictionary with keys {'x': [FILENAMES], 'y': [FILENAMES] } and loads the TIF filenames into an array.
        '''
        ...


def main(x):
    _test(x)

def _test(x):
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

    

if __name__ == "__main__":
    app.run(main)


