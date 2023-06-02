import os
from PIL import Image
import skvideo.io
import random
from moviepy.editor import *

shots_root = '/playpen-storage/mmiemon/datasets/OVSD/shots'
video_root = '/playpen-storage/mmiemon/datasets/OVSD/videos'
files = os.listdir(video_root)

random.shuffle(files)

for file in files:
    name = file.split('.')[0]
    if name == 'lord':
        continue
    print(name, file)
    dest = f'../data/OVSD/frames/{name}'
    if os.path.exists(dest):
        continue
    try:
        #videodata = skvideo.io.vread(f'{video_root}/{file}')
        clip = VideoFileClip(f'{video_root}/{file}')
        os.makedirs(dest)
    except:
        print('Cannot read file')
        continue
    with open(f'{shots_root}/{name}.txt', 'r') as f:
        shots = f.readlines()
    for cnt, s in enumerate(shots):
        a, b = s.strip().split()
        a, b = int(a), int(b)
        step = (b-a)/4
        for i in range(3):
            idx = int(a + (i + 1) * step)
            #frame = videodata[frame]
            frame = clip.get_frame(idx / clip.fps)
            frame = Image.fromarray(frame)
            shot = str(cnt).zfill(4)
            dest_file = f'{dest}/shot_{shot}_img_{i}.jpg'
            frame.save(dest_file)