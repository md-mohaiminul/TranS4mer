# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging
import os
import random

import einops
import ndjson
import numpy as np
import torch
from dataset.base import BaseDataset

class BBCDataset(BaseDataset):
    def __init__(self, cfg, mode, is_train):
        super(BBCDataset, self).__init__(cfg, mode, is_train)

        logging.info(f"Load Dataset: {cfg.DATASET}")
        if mode == "finetune" and not self.use_raw_shot:
            assert len(self.cfg.PRETRAINED_LOAD_FROM) > 0
            self.shot_repr_dir = os.path.join(
                self.cfg.FEAT_PATH, self.cfg.PRETRAINED_LOAD_FROM
            )

    def load_data(self):
        self.tmpl = "{}/shot_{}_img_{}.jpg"  # video_id, shot_id, shot_num
        if self.mode == "extract_shot":
            with open(
                os.path.join(self.cfg.ANNO_PATH, "annotator_0.ndjson"), "r"
            ) as f:
                self.anno_data = ndjson.load(f)

        elif self.mode == "finetune":
            if self.is_train:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "annotator_0.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)
                    #self.anno_data = self.anno_data[:len(self.anno_data) // 2]

                self.vidsid2label = {
                    f"{it['video_id']}_{it['shot_id']}": it["boundary_label"]
                    for it in self.anno_data
                }

            else:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "annotator_2.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)
                    #self.anno_data = self.anno_data[:len(self.anno_data) // 2]

            self.use_raw_shot = self.cfg.USE_RAW_SHOT
            if not self.use_raw_shot:
                self.tmpl = "{}/shot_{}.npy"  # video_id, shot_id

    def _get_mask(self, N: int):
        mask = np.zeros(N).astype(np.float)

        for i in range(N):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                mask[i] = 1.0

        if (mask == 0).all():
            # at least mask 1
            ridx = random.choice(list(range(0, N)))
            mask[ridx] = 1.0
        return mask

    def _getitem_for_extract_shot(self, idx: int):
        data = self.anno_data[
            idx
        ]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid = data["video_id"]
        sid = data["shot_id"]
        payload = {"vid": vid, "sid": sid}
        # video, place, audio, s = self.load_shot(vid, sid)

        #added later
        num_shot = data["num_shot"]
        sparse_idx, dense_idx = self.shot_sampler(int(sid), num_shot)
        video, place, audio = self.load_shot_list(vid, dense_idx)

        video = self.apply_transform(video)
        video = einops.rearrange(video, "(s k) c ... -> s k c ...", s=len(dense_idx))

        payload["video"] = video  # [s=1 k c h w]
        payload["place"] = place
        payload["audio"] = audio

        # other = self.load_other_modality(vid, sid)
        # payload["other"] = other  # [2048]

        assert "video" in payload
        return payload

    def _getitem_for_finetune(self, idx: int):
        data = self.anno_data[
            idx
        ]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid, sid = data["video_id"], data["shot_id"]
        num_shot = data["num_shot"]

        shot_idx = self.shot_sampler(int(sid), num_shot)

        if self.use_raw_shot:
            video, place, audio = self.load_shot_list(vid, shot_idx)
            video = self.apply_transform(video)
            if self.is_train:
                video = video.view(
                    len(shot_idx), 1, -1, 224, 224
                )  # the shape is [S,1,C,H,W]
            else:
                video = video.view(
                    len(shot_idx), 3, 3, 224, 224
                )  # the shape is [S,1,C,H,W]

        else:
            # _video = []
            # for sidx in shot_idx:
            #     shot_feat_path = os.path.join(
            #         self.shot_repr_dir, self.tmpl.format(vid, f"{sidx:04d}")
            #     )
            #     shot = np.load(shot_feat_path)
            #     shot = torch.from_numpy(shot)
            #     print(shot.shape)
            #     if len(shot.shape) > 1:
            #         shot = shot.mean(0)
            #
            #     _video.append(shot)
            # video = torch.stack(_video, dim=0)
            # _, place, audio = self.load_shot_list(vid, shot_idx, load_raw_video=False)

            shot_feat_path = os.path.join(self.shot_repr_dir, self.tmpl.format(vid, sid))
            shot = np.load(shot_feat_path)
            video = torch.from_numpy(shot)
            place = audio = []

        payload = {
            "idx": idx,
            "vid": vid,
            "sid": sid,
            "video": video,
            "place": place,
            "audio": audio,
            "label": abs(data["boundary_label"]),  # ignore -1 label.
        }

        return payload

    def _getitem_for_sbd_eval(self, idx: int):
        return self._getitem_for_finetune(idx)

    def __getitem__(self, idx: int):
        if self.mode == "extract_shot":
            return self._getitem_for_extract_shot(idx)

        elif self.mode == "pretrain":
            if self.is_train:
                return self._getitem_for_pretrain(idx)
            else:
                return self._getitem_for_knn_val(idx)

        elif self.mode == "finetune":
            if self.is_train:
                return self._getitem_for_finetune(idx)
            else:
                return self._getitem_for_sbd_eval(idx)
