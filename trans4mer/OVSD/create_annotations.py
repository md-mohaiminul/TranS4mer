import ndjson
import glob
import collections
import os

shots_root = '/playpen-storage/mmiemon/datasets/OVSD/shots'
scenes_root = '/playpen-storage/mmiemon/datasets/OVSD/scenes'
anno_root = '/playpen-storage/mmiemon/MovieNet/bassl/bassl/data/OVSD/anno'

shot_files = os.listdir(shots_root)

input = []
for file in shot_files:
    name = file.split('.')[0]
    if name == 'lord':
        continue
    scene_file = f'/playpen-storage/mmiemon/datasets/OVSD/scene_boundaries/{name}.txt'
    print(scene_file)
    with open(scene_file) as f:
        lines = f.readlines()[0].split(',')[:-1]
    boundaries = [int(b) for b in lines]
    print(boundaries)

    with open(f'{shots_root}/{name}.txt', 'r') as f:
        shots = f.readlines()
    num_shots = len(shots)        #change later

    for s in range(num_shots):
        dict = {}
        dict["video_id"] = name
        dict["shot_id"] = str(s).zfill(4)
        dict["num_shot"] = num_shots
        #dict["boundary_label"] = 1 if (s-1) in boundaries else 0
        if s in boundaries or (s+1) in boundaries:
            dict["boundary_label"] = 1
        else:
            dict["boundary_label"] = 0
        input.append(dict)

with open(f'{anno_root}/anno.test.ndjson', 'w') as f:
    ndjson.dump(input, f)




