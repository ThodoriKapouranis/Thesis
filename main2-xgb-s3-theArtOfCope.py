""" 
Hardcode and rewriting parts of the XGB training pipeline for scenario 3 
Because I have no idea whats wrong and I just need this one piece to work to progress
and im all out of ideas ha  - ha - ha -ha 
"""
import time
import numpy as np
import matplotlib
import rasterio
from xgboost import XGBClassifier
from DatasetHelpers.Dataset import create_dataset
from absl import app, flags
from DatasetHelpers.Preprocessing import PyRAT_rlf, PyRAT_sigma, frost_filter, lee_filter, box_filter

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


FLAGS = flags.FLAGS

flags.DEFINE_bool("debug", False, "Set logging level to debug")
flags.DEFINE_integer("scenario", 3, "Training data scenario. \n\t 1: Only co_event \n\t 2: coevent & preevent \n\t 3: coevent & preevent & coherence")
flags.DEFINE_string("filter", None, "None / lee")

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
flags.DEFINE_integer('batches', 4, 'Batches')

#Define model metadata
flags.DEFINE_string("savename", "xgb-test", "Name to use to save the model")

def main(x):
    batches = FLAGS.batches
    
    dataset = create_dataset(FLAGS)
    dataset.generate_batches(batches)
    sb = dataset.batches[0]

    def identity(x):
        return x 
    
    filter = identity

    filter = identity
    if FLAGS.filter == 'lee':
        filter = lee_filter
    if FLAGS.filter == 'box':
        filter = box_filter
    if FLAGS.filter == 'rfl':
        filter = PyRAT_rlf
    if FLAGS.filter == 'sigma':
        filter = PyRAT_sigma
    if FLAGS.filter == "frost":
        filter = frost_filter


    print(filter)
    
    # Do that training
    def load_data(batch:dict, scenario:int, filter:any):
        x = np.zeros(shape = (1, scenario*2))
        y =  np.zeros((1,1))

        for idx, scenes in enumerate(batch['x']):
            full_data = np.zeros(shape=(512*512, 1))
            for scene in scenes:
                data = rasterio.open(scene, 'r').read() # C, W, H
                data = filter(data)
                # S3 --> co-VV, co-VH, pre-VV, pre-VH, co-coh, pre-coh
                channels, width, height = data.shape
                data = np.reshape(data, (channels, width*height))
                data = np.transpose(data, (1,0))
                
                full_data = np.append(full_data, data, axis=1)
                # print("Current data:", data.shape)
                # print("Full data:", full_data.shape)
            
            # Remove zero channel from initialization
            full_data = full_data[:, 1:]
            x = np.append(x, full_data, axis=0)
        
        for idx, scenes in enumerate(batch['y']):
            
            # same as scene = scenes[0] because there will never be more than one target image.
            for scene in scenes:
                data = rasterio.open(scene, 'r').read()
                channels, width, height = data.shape
                data = np.reshape(data, (width*height, channels) )
                data = np.int32(data)
            
            y = np.append(y, data, axis=0)
            
        
        print(x.shape, y.shape)
        invalids = y[:,0] == -1
        y[:,0][invalids] = 0 # Set all -1 to 0 
        return x,y

    full_model = None
    
    for i in range(batches):
        model = XGBClassifier(use_label_encoder=False, tree_method = "hist", device = "cuda")
        model.verbosity = 0

        t1 = time.time()
        x, y = load_data(dataset.batches[i], FLAGS.scenario, filter=filter)

        print( f'Batch {i} finished loading in {time.time() - t1} seconds')
        
        if full_model != None: # Continue training with new batch
            model.fit(x, y, xgb_model=full_model.get_booster())
        else: 
            model.fit(x, y, xgb_model=None)
        
        full_model = model
    
    print("Finished training")
    # Done training
    full_model.save_model(f"Results/Models/{FLAGS.savename}.json")

        



if __name__ == "__main__":
    app.run(main)