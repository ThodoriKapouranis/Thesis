import os
import rasterio
import numpy as np
from tqdm import tqdm, trange
from absl import app, flags

S2 = '/workspaces/Thesis/10m_data/s2_labels'
S1_co = '/workspaces/Thesis/10m_data/s1_co_event_grd'
S1_pre = '/workspaces/Thesis/10m_data/s1_pre_event_grd'
hand_co = '/workspaces/Thesis/10m_data/coherence/hand_labeled/co_event'
hand_pre = '/workspaces/Thesis/10m_data/coherence/hand_labeled/pre_event'

is_tif_file = lambda x: True if x[-4:]==".tif" else False

FLAGS = flags.FLAGS
flags.DEFINE_bool("stats_label", False, "Give statistics of label dataset pixels")
flags.DEFINE_bool("stats_data", False, "Give statistics of dataset pixels")

def count_labels(directory:str):
    water = 0
    non_water = 0
    invalid = 0
    for file in tqdm(os.listdir(directory)):
        if is_tif_file(file) is False:
            continue

        data = rasterio.open(directory+'/'+file, 'r').read()
        water += len(data[data==1])
        non_water += len(data[data==0])
        invalid += len(data[data==-1])

    return water, non_water, invalid

def count_valid_data(directory:str):
    '''
    Goes through all files and grabs their max value.
    Invalid data is numpy nan
    '''
    invalid = 0
    total = 0
    file_list = os.listdir(directory)
    # max_hist = []
    pbar = trange(len(file_list))
    for i in pbar:
        file = file_list[i]
        if is_tif_file(file) is False:
            continue
        max = np.max(rasterio.open(directory+'/'+file).read())
        # max_hist.append(max)
        if max==0:
            invalid += 1
        # if np.isnan(max):
            # invalid += 1

        pbar.set_description(f'nan: {invalid}')
        total += 1
    
    return invalid, total

def display_label_stats(message, water, non_water, invalid):
    total = water + non_water + invalid
    print(f' {message} \n \
Water: {(100*water/total):.2f} Non_water: {(100 * non_water / total):.2f} Invalid : {(100*invalid/total):.2f}')

def display_data_stats(message, invalid, total):
    print(f"""{message} \n total \t nan \t nan % \n {total} \t {invalid} \t {(100 * invalid / total):.2f}""")

def main(a):
    if FLAGS.stats_label:
        for folder, msg in zip([S2, hand_co, hand_pre], ['s2', 'hand_co', 'hand_pre']):
            display_label_stats(msg, *count_labels(folder))
    
    if FLAGS.stats_data:
        for folder, msg in zip([S1_pre], ['s1_preevent']):
            invalid, total = count_valid_data(folder)

if __name__ == "__main__":
    app.run(main)