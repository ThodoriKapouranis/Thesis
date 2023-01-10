import os
import rasterio
import numpy as np
from tqdm import tqdm

S2 = '/workspaces/Thesis/10m_data/s2_labels'
S1_co = '/workspaces/Thesis/10m_data/s1_co_event_grd'
S1_pre = '/workspaces/Thesis/10m_data/s1_co_event_grd'

is_tif_file = lambda x: True if x[-4:]==".tif" else False

'''
Pixel Distribution of S2 Datasets
'''
water = 0
non_water = 0
invalid = 0
for file in tqdm(os.listdir(S2)):
    if is_tif_file(file) is False:
        continue

    data = rasterio.open(S2+'/'+file, 'r').read()
    water += len(data[data==1])
    non_water += len(data[data==0])
    invalid += len(data[data==-1])

total = water + non_water + invalid

print(f'Water: {100 * water/total} \n Non_water: {100 * non_water / total} \n Invalid : {100 * invalid / total}')