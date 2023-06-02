import ndjson
import glob

annotator = 'annotator_0'

input = []
for episode in range(1,12):
    episode = str(episode).zfill(2)
    print(episode)
    scene_files = glob.glob(f'/playpen-storage/mmiemon/datasets/BBC/annotations/scenes/{annotator}/*.txt')
    scene_file = [s for s in scene_files if episode in s][0]
    print(scene_file)
    with open(scene_file) as f:
        lines = f.readlines()[0].strip().split(',')
    boundaries = [int(b) for b in lines]
    num_shots = boundaries[-1]
    print(len(lines), num_shots)

    for s in range(num_shots):
        dict = {}
        ep = episode
        dict["video_id"] = ep
        dict["shot_id"] = str(s).zfill(4)
        dict["num_shot"] = num_shots
        dict["boundary_label"] = 1 if (s+1) in boundaries else 0
        input.append(dict)

with open(f'data/BBC/anno/{annotator}.ndjson', 'w') as f:
    ndjson.dump(input, f)


# input = []
# for episode in range(1,11):
#     episode = str(episode).zfill(2)
#     print(episode)
#     scene_filess = glob.glob('/playpen-storage/mmiemon/datasets/BBC/annotations/scenes/*/*.txt')
#     scene_files = [s for s in scene_filess if episode in s]
#     print(scene_files)
#     boundaries = []
#     for scene_file in scene_files:
#         with open(scene_file) as f:
#             lines = f.readlines()[0].strip().split(',')
#         bb = [int(b) for b in lines]
#         boundaries += bb
#
#     num_shots = boundaries[-1]
#     print(len(lines), num_shots)
#
#     for s in range(num_shots):
#         dict = {}
#         ep = episode
#         dict["video_id"] = ep
#         dict["shot_id"] = str(s).zfill(4)
#         dict["num_shot"] = num_shots
#         dict["boundary_label"] = 1 if (s+1) in boundaries else 0
#         input.append(dict)
#
# with open('data/BBC/anno/annotator_all.ndjson', 'w') as f:
#     ndjson.dump(input, f)

