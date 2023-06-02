import glob
import random
import os

from scenedetect import detect, ContentDetector
# scene_list = detect('/playpen-storage/mmiemon/datasets/BBC/videos/bbc_01.mp4', ContentDetector())

files = glob.glob('/playpen-storage/mmiemon/datasets/OVSD/videos/*')
random.shuffle(files)


dest = '/playpen-storage/mmiemon/datasets/OVSD/shots'

for cnt, file in enumerate(files):
    file_name = (file.split('/')[-1]).split('.')[0]
    dest_file = f'{dest}/{file_name}.txt'
    if not os.path.exists(dest_file):
        print(cnt, file, file_name)
        scene_list = detect(file, ContentDetector())
        with open(dest_file, 'w') as f:
            for i, scene in enumerate(scene_list):
                s = str(scene[0].get_frames()) + ' ' + str(scene[1].get_frames()) + '\n'
                f.write(s)
                #print(scene[0].get_frames(), scene[1].get_frames())
                # print('Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                #     i+1,
                #     scene[0].get_timecode(), scene[0].get_frames(),
                #     scene[1].get_timecode(), scene[1].get_frames(),))
