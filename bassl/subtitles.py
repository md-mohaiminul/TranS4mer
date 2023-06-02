import pysrt
import json
import ndjson

id = 'tt0052357'    #tt0052357

subs = pysrt.open(f'data/movienet/features/subtitle/{id}.srt', encoding='iso-8859-1')
# parts = subs.slice(starts_after={'seconds': 0}, ends_before={'seconds': 400})
# print(parts.text)

# for sub in subs:
#     #print(sub.text)
#     print(sub.start, sub.start.hours, sub.start.minutes, sub.start.seconds, sub.start.milliseconds)
#     break
# print(len(subs))

video_info = '/playpen-storage/mmiemon/datasets/movienet/video_info.json'
with open(video_info, 'r') as f:
    video_info = json.load(f)
print(video_info[id])
fps = float(video_info[id]['fps'])

with open("data/movienet/anno/anno.test.ndjson", "r") as f:
    anno = ndjson.load(f)
for item in anno:
    if item['video_id']==id:
        print(item['shot_id'], item['num_shot'])
        break

shot_detection = f'/playpen-storage/mmiemon/datasets/movienet/shot_detection_1K/{id}.txt'

with open(shot_detection, 'r') as fp:
    lines = fp.readlines()
print(len(lines))

with open(f'subtitle_examples/{id}.txt', 'w') as f:
    for count, line in enumerate(lines):
        #print('x', line.strip().split())
        start = float(line.strip().split()[0])/fps
        end = float(line.strip().split()[1])/fps
        parts = subs.slice(starts_after={'seconds': start}, ends_before={'seconds': end})
        print('Shot :', count)
        print(parts.text)
        print()

        f.write('Shot :'+str(count)+'\n')
        f.write(parts.text)
        f.write('\n\n')

f.close()



