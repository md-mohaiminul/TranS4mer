import pickle

import ndjson
import os
import glob
import numpy as np

# file = '/playpen-storage/mmiemon/MovieNet/bassl/bassl/data/movienet/anno/anno.trainvaltest.ndjson'
#
# with open(file, "r") as f:
#     anno_data = ndjson.load(f)
#
# for item in anno_data:
#     print(item.keys())
#     break

source = '/playpen-storage/mmiemon/datasets/movienet/audio_feat_1K'
dest = '/playpen-storage/mmiemon/datasets/movienet/audio_feat_mean_1K'

for file in os.listdir(source):
    print(file)
    with open(f'{dest}/{file}', 'rb') as f:
        item = pickle.load(f)
    for key in item:
        print(key, item[key].shape)
    #     item[key] = np.mean(item[key], axis=0)
    #     #print(key, item[key].shape)
    # with open(f'{dest}/{file}', 'wb') as f:
    #     pickle.dump(item, f)