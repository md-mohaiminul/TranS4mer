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

class MovieNetDataset(BaseDataset):
    def __init__(self, cfg, mode, is_train):
        super(MovieNetDataset, self).__init__(cfg, mode, is_train)

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
                os.path.join(self.cfg.ANNO_PATH, "anno.trainvaltest.ndjson"), "r"
            ) as f:
                self.anno_data = ndjson.load(f)

        elif self.mode == "pretrain":
            if self.is_train:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.pretrain.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)
                    #self.anno_data = self.anno_data[:len(self.anno_data)//2]
                # with open(
                #         os.path.join(self.cfg.ANNO_PATH, "anno.test.ndjson"), "r"
                # ) as f:
                #     self.anno_data = ndjson.load(f)
            else:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.test.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)
                    #self.anno_data = self.anno_data[:len(self.anno_data)//2]

        elif self.mode == "finetune":
            if self.is_train:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.train.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)
                    #self.anno_data = self.anno_data[:len(self.anno_data) // 2]

                self.vidsid2label = {
                    f"{it['video_id']}_{it['shot_id']}": it["boundary_label"]
                    for it in self.anno_data
                }

            else:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.test.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)
                    #self.anno_data = self.anno_data[:len(self.anno_data) // 2]

            self.use_raw_shot = self.cfg.USE_RAW_SHOT
            if not self.use_raw_shot:
                self.tmpl = "{}/shot_{}.npy"  # video_id, shot_id

    def _getitem_for_pretrain(self, idx: int):
        data = self.anno_data[
            idx
        ]  # contain {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid = data["video_id"]
        sid = data["shot_id"]
        num_shot = data["num_shot"]
        payload = {"idx": idx, "vid": vid, "sid": sid}

        if self.sampling_method in ["instance", "temporal"]:
            # This is for two shot-level pre-training baselines:
            # 1) SimCLR (instance) and 2) SimCLR (temporal)
            keyframes, nshot = self.load_shot(vid, sid)
            view1 = self.apply_transform(keyframes)
            view1 = einops.rearrange(view1, "(s k) c ... -> s (k c) ...", s=nshot)

            new_sid = self.shot_sampler(int(sid), num_shot)
            if not new_sid == int(sid):
                keyframes, nshot = self.load_shot(vid, sid)
            view2 = self.apply_transform(keyframes)
            view2 = einops.rearrange(view2, "(s k) c ... -> s (k c) ...", s=nshot)

            # video shape: [nView=2,S,C,H,W]
            video = torch.stack([view1, view2])
            payload["video"] = video

        elif self.sampling_method in ["shotcol", "bassl+shotcol", "bassl"]:
            sparse_method = "edge" if self.sampling_method == "bassl" else "edge+center"
            sparse_idx_to_dense, dense_idx = self.shot_sampler(
                int(sid), num_shot, sparse_method=sparse_method
            )
            #sparse_idx_to_dense [0 16] dense_idx [s ... s+16]

            # load densely sampled shots (=S_n)
            _dense_video, dense_place, dense_audio = self.load_shot_list(vid, dense_idx)
            dense_video = self.apply_transform(_dense_video)
            if self.cfg.TRAIN.USE_DOUBLE_KEYFRAME:
                dense_video = dense_video.view(len(dense_idx), 2, 3, 224, 224)
                # [nDenseShot,C,H,W] corresponding to S_n   [17, 2, 3, 224, 224]
            else:
                dense_video = dense_video.view(len(dense_idx), -1, 224, 224)
                # [nDenseShot,C,H,W] corresponding to S_n   [17, 3, 224, 224]

            # fetch sparse sequence from loaded dense sequence (=S_n^{slow})
            if self.cfg.TRAIN.USE_DOUBLE_KEYFRAME:
                double_idx = []
                for i in sparse_idx_to_dense:
                    double_idx.append(2*i)
                    double_idx.append(2*i+1)
                _sparse_video = [_dense_video[idx] for idx in double_idx]
                sparse_video = self.apply_transform(_sparse_video)
                sparse_video = sparse_video.view(len(sparse_idx_to_dense), 2, 3, 224, 224)
            else:
                _sparse_video = [_dense_video[idx] for idx in sparse_idx_to_dense]
                sparse_video = self.apply_transform(_sparse_video)
                sparse_video = sparse_video.view(len(sparse_idx_to_dense), -1, 224, 224)
            # [nSparseShot,C,H,W] corresponding to S_n^{slow}    [2, 3, 224, 224]
            # if not using temporal modeling, video shape is [T=nsparse+ndense, S=1, C=3, H, W]
            video = torch.cat([sparse_video, dense_video], dim=0) #[19, 3, 224, 224]
            if not self.cfg.TRAIN.USE_DOUBLE_KEYFRAME:
                video = video[:, None, :]  # [T,S=1,C,H,W]  [19, 1, 3, 224, 224]

            #added later
            # dense_other = self.load_other_modality_list(vid, dense_idx)
            # dense_other= torch.from_numpy(dense_other)
            # sparse_other = dense_other[sparse_idx_to_dense]
            # other = torch.cat([sparse_other, dense_other], dim=0)  #[19, 2048]

            if 'PLACE' in self.cfg.OTHER_MODALITY.TYPE:
                dense_place = torch.from_numpy(dense_place)
                sparse_place = dense_place[sparse_idx_to_dense]
                place = torch.cat([sparse_place, dense_place], dim=0)  #[19, 2048]
                payload["place"] = place

            if 'AUDIO' in self.cfg.OTHER_MODALITY.TYPE:
                dense_audio = torch.from_numpy(dense_audio)
                sparse_audio = dense_audio[sparse_idx_to_dense]
                audio = torch.cat([sparse_audio, dense_audio], dim=0)  # [19, 2048]
                payload["audio"] = audio

            #video = dense_video[:, None, :]  # [T,S=1,C,H,W]  [17, 1, 3, 224, 224]   #added later
            payload["video"] = video
            payload["sparse_idx"] = sparse_idx_to_dense  # to compute nsparse
            payload["dense_idx"] = dense_idx
            payload["mask"] = self._get_mask(len(dense_idx))  # for MSM pretext task

        else:
            raise ValueError

        assert "video" in payload
        return payload

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

    def _getitem_for_knn_val(self, idx: int):
        data = self.anno_data[
            idx
        ]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid = data["video_id"]
        sid = data["shot_id"]
        payload = {
            "global_video_id": data["global_video_id"],
            "sid": sid,
            "invideo_scene_id": data["invideo_scene_id"],
            "global_scene_id": data["global_scene_id"],
        }
        #video, place, audio, s = self.load_shot(vid, sid)

        # added later
        num_shot = data["num_shot"]
        sparse_idx, dense_idx = self.shot_sampler(int(sid), num_shot)
        video, place, audio = self.load_shot_list(vid, dense_idx)

        video = self.apply_transform(video)
        video = einops.rearrange(video, "(s k) c ... -> s k c ...", s=len(dense_idx))
        payload["video"] = video
        payload["place"] = place
        payload["audio"] = audio

        #added later
        # num_shot = data["num_shot"]
        # sparse_method = "edge" if self.sampling_method == "bassl" else "edge+center"
        # sparse_idx_to_dense, dense_idx = self.shot_sampler(
        #     int(sid), num_shot, sparse_method=sparse_method
        # )
        # _dense_video, dense_place, dense_audio = self.load_shot_list(vid, dense_idx)
        # dense_video = self.apply_transform(_dense_video)
        # dense_video = dense_video.view(
        #     len(dense_idx), 3, 3, 224, 224
        # )
        # #video = dense_video[:, None, :]  # [T,S=1,C,H,W]  [17, 1, 3, 224, 224]   #added later
        # payload["video"] = dense_video

        # other = self.load_other_modality(vid, sid)
        # payload["other"] = other  # [2048]

        assert "video" in payload
        return payload

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
