import json
import ndjson
import pickle

import numpy as np
from sklearn.metrics import average_precision_score

class SequenceShotSampler:
    """ This is for bassl or shotcol at pre-training stage """
    def __init__(self, neighbor_size: int, neighbor_interval: int):
        self.interval = neighbor_interval
        self.window_size = neighbor_size * self.interval  # temporal coverage

    def __call__(
        self, center_sid: int, total_num_shot: int, sparse_method: str = "edge"
    ):
        """
        Args:
            center_sid: index of center shot
            total_num_shot: last index of shot for given video
            sparse_stride: stride to sample sparse ones from dense sequence
                    for curriculum learning
        """

        dense_shot_idx = center_sid + np.arange(
            -self.window_size, self.window_size + 1, self.interval
        )  # total number of shots = 2*neighbor_size+1

        if dense_shot_idx[0] < 0:
            # if center_sid is near left-side of video, we shift window rightward
            # so that the leftmost index is 0
            dense_shot_idx -= dense_shot_idx[0]
        elif dense_shot_idx[-1] > (total_num_shot - 1):
            # if center_sid is near right-side of video, we shift window leftward
            # so that the rightmost index is total_num_shot - 1
            dense_shot_idx -= dense_shot_idx[-1] - (total_num_shot - 1)

        # to deal with videos that have smaller number of shots than window size
        dense_shot_idx = np.clip(dense_shot_idx, 0, total_num_shot)

        if sparse_method == "edge":
            # in this case, we use two edge shots as sparse sequence
            sparse_stride = len(dense_shot_idx) - 1
            sparse_idx_to_dense = np.arange(0, len(dense_shot_idx), sparse_stride)
        elif sparse_method == "edge+center":
            # in this case, we use two edge shots + center shot as sparse sequence
            sparse_idx_to_dense = np.array(
                [0, len(dense_shot_idx) - 1, len(dense_shot_idx) // 2]
            )
        # sparse_shot_idx = dense_shot_idx[sparse_idx_to_dense]

        # shot_idx = [sparse_shot_idx, dense_shot_idx]
        shot_idx = [sparse_idx_to_dense, dense_shot_idx]
        return shot_idx

# with open('/playpen-storage/mmiemon/MovieNet/bassl/bassl/outputs/dtw_preds.json', 'r') as f:
#     preds = json.load(f)

with open('/playpen-storage/mmiemon/MovieNet/bassl/bassl/outputs/slope_clustering_preds.pickle', 'rb') as handle:
    preds = pickle.load(handle)

with open('data/movienet/anno/anno.test.ndjson', 'rb') as f:
    anno = ndjson.load(f)

sampler = SequenceShotSampler(neighbor_size=12, neighbor_interval=1)

# x = sampler(10, 100)
#
# print(x)

# all_gt = {}
# for item in anno:
#     vid = item['video_id']
#     sid = int(item['shot_id'])
#     label = abs(item['boundary_label'])
#     key = f'{vid}_{sid}'
#     all_gt[key] = label
#     #print(key, all_gt[key])
#
# found = 0
# total = 0
# for item in anno:
#     vid = item['video_id']
#     sid = int(item['shot_id'])
#     num_shot = int(item['num_shot'])
#     _, shots = sampler(sid, num_shot)
#     for s in shots:
#         k = f'{vid}_{s}'
#         if k not in all_gt:
#             print(k)
#             continue
#         if all_gt[k]==1:
#             found += 1
#             break
#     total += 1
#
# print(found, len(anno), total, found/total)
# for item in preds:
#     print(preds[item])
#     print(all_gt[item])
#     break

movies = []
tp = fp = tn = fn = 0

gts = []
ps = []
r1 = r0 = c1 = c0 = 0
for item in anno:
    vid = item['video_id']
    sid = item['shot_id']
    label = abs(item['boundary_label'])
    key = f'{vid}_{sid}'
    if key not in preds:
        continue
    gts.append(label)

    # if label==1:
    #     if preds[key] == 8:
    #         r1 += 1
    #     c1 += 1
    # else:
    #     if preds[key] != 8:
    #         r0 += 1
    #     c0 += 1

    # if preds[key] == 8:
    #     ps.append(1)
    # else:
    #     ps.append(0)

    if preds[key] == 8:
        if label == 1:
            tp += 1
        else:
            fp += 1
    else:
        if label==0:
            tn += 1
        else:
            fn += 1
#     movies.append(vid)
#
# print(r1/c1, (r1+r0)/(c1+c0))

precision = tp/(tp+fp)
accuracy = (tp+tn)/(tp+fp+tn+fn)

print(precision, accuracy)
# ps = np.array(ps)
# gts = np.array(gts)
# precision = average_precision_score(gts, ps)
#
# print(precision)


