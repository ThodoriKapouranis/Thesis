import time
import numpy as np
import rasterio
import xgboost as xgb
from dataclasses import dataclass, field
from absl import app, flags

@dataclass
class Batched_XGBoost:
    model: any = field(init=False)
    
    def train_in_batches(self, batches:dict):
        '''
        This function trains xgboost models in batches in order to fit in the GPU's memory.
        The files are only loaded in memory once they are needed for training.
        
        Arguments:
        --  batches : a dictionary holding 'x' and 'y' keys that map to a list of training files names for every chip

        Returns:
        --  model   :   Trained XGBoost model using the xgboost library
        --?  hist    :   training info
        '''
        full_model = None

        for batch_idx in batches.keys():
            
            batch_model = xgb.XGBClassifier(use_label_encoder=False, tree_method='gpu_hist')
            batch_model.verbosity = 3
            batch_model.max_depth = 6
            batch_model.learning_rate = 0.3
            batch_model.n_estimators = 1
            
            t1 = time.time()
            x, y = self.__load_data(batches[batch_idx])
            print( f'Batch {batch_idx} finished loading in {time.time() - t1} seconds')
            
            print("Starting training...")
            t1 = time.time()
            batch_model.fit(x, y, verbose=True, xgb_model=full_model)
            print(f'Finished Training w/ batch {batch_idx} in {time.time() - t1} seconds')
            full_model = batch_model
        
        self.model =  full_model

    def __remap_labels(self, chip):
        # Incoming label chip should be size (1, 512x512)
        invalids = chip[0][:] == -1
        chip[0][invalids] = 0
        return chip
    
    # Better solution is to use a map to read the file names and replace with the squeezed data?
    def __load_data(self, batch:dict):
        '''
        Takes a dictionary with keys {'x': [FILENAMES], 'y': [FILENAMES] } and loads the TIF filenames into an array.
        '''

        channels = 2 # Do we keep channels??? Either way this needs to be a FLAG.scenario option
        x = np.zeros(shape = (len(batch['x']), channels*512*512))
        y =  np.zeros(shape = (len(batch['y']), 512*512))

        for idx, scenes in enumerate(batch['y']):
            for scene in scenes:
                data = rasterio.open(scene, 'r').read()
                channels, width, height = data.shape
                data = np.reshape(data, (channels, width*height) )
                data = self.__remap_labels(data)
                y[idx,:] = data
        
        for idx, scenes in enumerate(batch['x']):
            scenes = scenes[0]
            for scene in scenes:
                data = rasterio.open(scene, 'r').read() # C, W, H
                channels, width, height = data.shape
                data = data.flatten()
                x[idx,:] = data
        
        print(x.shape, y.shape)
        return x, y

def main(x):
    _test(x)

def _test(x):
    from DatasetHelpers.Dataset import Dataset, create_dataset, index_dataset

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
    batches = dataset.generate_batches(20)
    model = Batched_XGBoost()
    model.train_in_batches(batches)

if __name__ == "__main__":
    app.run(main)


