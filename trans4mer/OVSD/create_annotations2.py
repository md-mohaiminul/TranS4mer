import os
import ndjson

shots_root = '/playpen-storage/mmiemon/datasets/OVSD/shots'
scenes_root = '/playpen-storage/mmiemon/datasets/OVSD/scenes'
anno_root = '/playpen-storage/mmiemon/MovieNet/bassl/bassl/data/OVSD/anno'

shot_files = os.listdir(shots_root)

input = []
for file in shot_files:
    name = file.split('.')[0]
    if name == 'lord':
        continue
    print(name, file)
    with open(f'{scenes_root}/{name}.txt', 'r') as f:
        scenes = f.readlines()
    scene_boundaries = []
    for s in scenes:
        scene_boundaries.append(int(s.split()[-1]))
    print(scene_boundaries)

    with open(f'{shots_root}/{name}.txt', 'r') as f:
        shots = f.readlines()
    cur = 0
    for cnt, s in enumerate(shots):
        if cnt==(len(shots)-1):
            continue
        dict = {}
        dict["video_id"] = name
        dict["shot_id"] = str(cnt+1).zfill(4)
        dict["num_shot"] = len(shots)
        shot_boundary = int(s.split()[-1])
        if shot_boundary < scene_boundaries[cur]:
            dict["boundary_label"] = 0
        else:
            dict["boundary_label"] = 1
            cur += 1
            if cur>= len(scene_boundaries)-1:
                break
        input.append(dict)
        print(dict)

with open('/playpen-storage/mmiemon/MovieNet/bassl/bassl/data/OVSD/anno/anno.test.ndjson', 'w') as f:
    ndjson.dump(input, f)

