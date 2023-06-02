# import torch
# from tslearn import metrics
import json

import numpy as np
# import pickle
# import torch.nn as nn
# import ndjson, json
# import random
import pytorch_lightning as pl

# def _compute_dtw_path(s_emb, d_emb):
#     """ compute alignment between two sequences using DTW """
#     cost = (
#         (1 - torch.bmm(s_emb, d_emb.transpose(1, 2)))
#             .cpu()
#             .numpy()
#             .astype(np.float32)
#     )  # shape: [b n_sparse n_dense]
#     dtw_path = []
#     for bi in range(cost.shape[0]):
#         _path, _ = metrics.dtw_path_from_metric(cost[bi], metric="precomputed")
#         dtw_path.append(np.asarray(_path))  # [n_dense 2]
#
#     return dtw_path
#
# with open ('/playpen-storage/mmiemon/datasets/movienet/place_feat_1K/tt0209463.pkl', 'rb') as f:
#     place = pickle.load(f)
# print(place['0000'].shape, place['0000'].dtype)
#
# start = 50
# x = []
# for idx in range(start, start+17):
#     #scene = str(idx).zfill(4)
#     x.append(place[f"{idx:04d}"])
#
# x = np.stack(x)
# y = np.stack((x[0], x[16]), axis=0)
#
# x = torch.unsqueeze(torch.from_numpy(x), dim=0)
# y = torch.unsqueeze(torch.from_numpy(y), dim=0)
#
# head = nn.Linear(2048, 128)
#
# x = head(x).detach()
# y = head(y).detach()
#
# print(x.shape, y.shape)
#
# dtw_path = _compute_dtw_path(x, y)
#
# print(len(dtw_path))
# print(dtw_path)

# s_emb = torch.randn(32, 17, 128)
# # d_emb = torch.stack((s_emb[:,0,:],s_emb[:,16,:]), dim=1)
# d_emb = torch.randn(32, 2, 128)
# print(s_emb.shape, d_emb.shape)

# head_shot_repr = torch.randn(32, 19, 128)
# s_emb, d_emb = torch.split(head_shot_repr, [2, 17], dim=1)
# print(s_emb.shape, d_emb.shape)
#
# s_emb = torch.randn(32, 17, 128)
# d_emb = torch.randn(32, 2, 128)
#
# dtw_path = _compute_dtw_path(s_emb, d_emb)
#
# print(len(dtw_path))
# print(dtw_path[0])

# class audnet(nn.Module):
#     def __init__(self):
#         super(audnet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(2,1), padding=0)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=(1,3))
#
#         self.conv2 = nn.Conv2d(64, 192, kernel_size=(3,3), stride=(2,1), padding=0)
#         self.bn2 = nn.BatchNorm2d(192)
#         self.relu2 = nn.ReLU(inplace=True)
#         # self.pool2 = nn.MaxPool2d(kernel_size=(1,3))
#
#         self.conv3 = nn.Conv2d(192, 384, kernel_size=(3,3), stride=(2,1), padding=0)
#         self.bn3 = nn.BatchNorm2d(384)
#         self.relu3 = nn.ReLU(inplace=True)
#
#         self.conv4 = nn.Conv2d(384, 256, kernel_size=(3,3), stride=(2,2), padding=0)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.relu4 = nn.ReLU(inplace=True)
#
#         self.conv5 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(2,2), padding=0)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.relu5 = nn.ReLU(inplace=True)
#         self.pool5 = nn.MaxPool2d(kernel_size=(2,2))
#
#         self.conv6 = nn.Conv2d(256, 512, kernel_size=(3,2), padding=0)
#         self.bn6 = nn.BatchNorm2d(512)
#         self.relu6 = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(512, 512)
#
#     def forward(self, x):  # [bs,1,257,90]
#         x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
#         x = self.relu2(self.bn2(self.conv2(x)))
#         x = self.relu3(self.bn3(self.conv3(x)))
#         x = self.relu4(self.bn4(self.conv4(x)))
#         x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
#         x = self.relu6(self.bn6(self.conv6(x)))
#         x = x.squeeze()
#         out = self.fc(x)
#         return out

# m = '/playpen-storage/mmiemon/datasets/movienet/audio_feat_1K/tt0092699.pkl'
# with open(m, 'rb') as f:
#     x = pickle.load(f)
#
# for k in x:
#     print(k, x[k].shape)

# with open("data/movienet/anno/anno.val.ndjson", "r") as f:
#     anno = ndjson.load(f)
#
# for item in anno:
#     print(item)
#     break

# import ndjson
#
# with open(("data/BBC/anno/annotator_0.ndjson"), "r") as f:
#     anno_data = ndjson.load(f)
#     # self.anno_data = self.anno_data[:len(self.anno_data) // 2]
# print(anno_data)
#
# vidsid2label = {
#     f"{it['video_id']}_{it['shot_id']}": it["boundary_label"]
#     for it in anno_data
# }
#
# print(vidsid2label)


# import pandas as pd
# df = pd.read_csv('/playpen-storage/mmiemon/datasets/movienet/title.basics.tsv', sep='\t', index_col='tconst')
# print(df.head())
#
# print(df.loc['tt0000001']['primaryTitle'])

ann = 'data/movienet/scene318/meta/scene_movie318.json'
with open(ann, 'r') as f:
    ann = json.load(f)
print(ann['tt0068646'])

# for k in ann.keys():
#     print(k, df.loc[k]['primaryTitle'])
# for movie in ann:
#     print(ann)
#     break

# tt0068646 The Godfather