import time
import numpy as np
import rasterio
from xgboost import XGBClassifier
from dataclasses import dataclass, field
from absl import app, flags

XGB_POS_WEIGHT = 6.7233518222

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
            
            batch_model = XGBClassifier(use_label_encoder=False, tree_method='gpu_hist', scale_pos_weight=XGB_POS_WEIGHT)
            batch_model.verbosity = 3
            
            t1 = time.time()
            x, y = self.__load_data(batches[batch_idx])
            print( f'Batch {batch_idx} finished loading in {time.time() - t1} seconds')
            
            print("Starting training...")
            t1 = time.time()
            batch_model.fit(x, y, verbose=True, xgb_model=full_model)
            print(f'Finished Training w/ batch {batch_idx} in {time.time() - t1} seconds')
            full_model = batch_model
        
        self.model =  full_model

    def predict_in_batches(self, batches:dict):
        ...
        predictions = []
        truth = []
        for batch_idx in batches.keys():
            t1 = time.time()
            x, y = self.__load_data(batches[batch_idx])
            print( f'Batch {batch_idx} finished loading in {time.time() - t1} seconds')
            
            print("Starting batch prediction...")
            t1 = time.time()
            batch_pred = self.model.predict(x)

            predictions.append(batch_pred)
            truth.append(y)

            print(f'Predictions: {len(predictions)}, Truth: {len(truth)}')
        return predictions, truth

    def load_model(self, path):
        # TODO I dont think scale_pos_weight is necessary here because this model will not be used for training
        self.model = XGBClassifier(use_label_encoder=False, tree_method='gpu_hist', scale_pos_weight=XGB_POS_WEIGHT)
        self.model.load_model(path)

    def __remap_labels(self, chip):
        # Incoming label chip should be size (512x512, 1)
        invalids = chip[:,0] == -1
        chip[:,0][invalids] = 0
        return chip
    
    # Better solution is to use a map to read the file names and replace with the squeezed data?
    def __load_data(self, batch:dict):
        '''
        Takes a dictionary with keys {'x': [FILENAMES], 'y': [FILENAMES] } and loads the TIF filenames into an array.

        param X_train : 2D- ndarray with shape ( num_pix , num_feat ) with input features
        param Y_train : 2D- ndarray with shape ( num_pix ,) with labels

        '''
        channels = 2

        if batch['x'].shape[1] == 1:
            channels = 2
        elif batch['x'].shape[1] == 2:
            channels = 4
        elif batch['x'].shape[1] == 4:  
            channels = 6

        print(f"Expecting channel size: {channels}")
        x = np.zeros(shape = (1, channels))
        y =  np.zeros( (1,1) )

        # Batch comes in as (samples, file_urls)
        # So for S1, S2, S3 Respectively : (N,1), (N,2), (N,4)
        for idx, scenes in enumerate(batch['y']):
            # same as scene = scenes[0] because there will never be more than one target image.
            for scene in scenes:
                data = rasterio.open(scene, 'r').read()
                channels, width, height = data.shape
                data = np.reshape(data, (width*height, channels) )
                data = np.int32(data)
                data = self.__remap_labels(data)
                y = np.append(y, data, axis=0)
        
        print("Finished loading targets ...")

        for idx, scenes in enumerate(batch['x']):
            # To be completed by compiling all the sample's scenes in this variable
            full_data = np.zeros(shape=(512*512, 1)) 
            
            for scene in scenes:
                data = rasterio.open(scene, 'r').read() # C, W, H
                channels, width, height = data.shape
                data = np.reshape(data, (channels, width*height))
                data = np.transpose(data, (1,0))    
                full_data = np.append(full_data, data, axis=1)
            
            # Ditch first zero generated channel
            full_data = full_data[:,1:]
            x = np.append(x, full_data, axis=0)
        
        print("Finished loading data ...")

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


