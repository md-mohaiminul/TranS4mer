import skvideo.io
from moviepy.editor import *
import os
import glob
from PIL import Image

for episode in range(1, 12):
    episode = str(episode).zfill(2)
    print(episode)
    videos = glob.glob('/playpen-storage/mmiemon/datasets/BBC/videos/*.mp4')
    video = [v for v in videos if episode in v][0]
    print(video)
    videodata = skvideo.io.vread(video)
    print(videodata.shape)

    shot_files = glob.glob('/playpen-storage/mmiemon/datasets/BBC/annotations/shots/*.txt')
    shot_file = [s for s in shot_files if episode in s][0]
    print(shot_file)
    with open(shot_file) as f:
        lines = f.readlines()

    for shot, line in enumerate(lines):
        a, b = line.strip().split()
        a, b = int(a), int(b)
        #step = (b-a)//4
        step = (b - a) / 11
        dest = f'data/BBC/frames_10/{episode}'
        if not os.path.exists(dest):
            os.makedirs(dest)
        for i in range(10):
            frame = int(a + (i+1)*step)
            frame = videodata[frame]
            frame = Image.fromarray(frame)
            shot = str(shot).zfill(4)
            dest_file = f'{dest}/shot_{shot}_img_{i}.jpg'
            frame.save(dest_file)
