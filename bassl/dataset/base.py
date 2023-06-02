# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import copy
import logging
import os
import random
import pickle

import dataset.sampler as sampler
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
from transform import get_transform


class BaseDataset(Dataset):
    def __init__(self, cfg, mode, is_train):
        self.cfg = cfg
        self.mode = mode
        self.is_train = is_train

        self.use_single_keyframe = cfg.TRAIN.USE_SINGLE_KEYFRAME
        self.num_keyframe = cfg.TRAIN.NUM_KEYFRAME
        self.use_double_keyframe = cfg.TRAIN.USE_DOUBLE_KEYFRAME

        self.load_data()
        self.init_transform(cfg)
        self.init_sampler(cfg)

    def __getitem__(self, idx: int):
        raise NotImplementedError()

    def __len__(self):
        return len(self.anno_data)

    def load_image(self, path):
        return pil_loader(path)

    def load_shot_keyframes(self, path):
        shot = None
        if self.use_double_keyframe:
            indices = random.sample([0,1,2], 2)
            indices.sort()
            shot = [pil_loader(path.format(i)) for i in indices]
        elif self.is_train and self.use_single_keyframe:
            # load one randomly sampled keyframe
            shot = [pil_loader(path.format(random.randint(0, self.num_keyframe - 1)))]
        else:
            # load all keyframes
            shot = [pil_loader(path.format(i)) for i in range(self.num_keyframe)]
        assert shot is not None
        return shot

    def load_shot_list(self, vid, shot_idx, load_raw_video=True):
        shot_list = []
        cache = {}
        if load_raw_video:
            for sidx in shot_idx:
                vidsid = f"{vid}_{sidx:04d}"
                if vidsid in cache:
                    shot = cache[vidsid]
                else:
                    shot_path = os.path.join(
                        self.cfg.IMG_PATH, self.tmpl.format(vid, f"{sidx:04d}", "{}")
                    )
                    shot = self.load_shot_keyframes(shot_path)
                    cache[vidsid] = shot
                shot_list.extend(shot)

        place_list = []
        if 'PLACE' in self.cfg.OTHER_MODALITY.TYPE:
            vid_path = os.path.join(self.cfg.OTHER_MODALITY.PLACE_PATH, f"{vid}.pkl")
            with open(vid_path, 'rb') as f:
                place = pickle.load(f)
            for sidx in shot_idx:
                place_list.append(place[f"{sidx:04d}"])
            place_list = np.stack(place_list)

        audio_list = []
        if 'AUDIO' in self.cfg.OTHER_MODALITY.TYPE:
            vid_path = os.path.join(self.cfg.OTHER_MODALITY.AUDIO_PATH, f"{vid}.pkl")
            with open(vid_path, 'rb') as f:
                audio = pickle.load(f)
            for sidx in shot_idx:
                key = f"{sidx:04d}"
                if key in audio:
                    shot = audio[key]
                else:
                    shot = np.zeros(90, dtype=float)
                audio_list.append(shot)
            audio_list = np.stack(audio_list)

        return shot_list, place_list, audio_list

    # def load_other_modality_list(self, vid, shot_idx):
    #     shot_list = []
    #     cache = {}
    #     if vid in cache:
    #         place = cache[vid]
    #     else:
    #         vid_path = os.path.join(self.cfg.OTHER_MODALITY.PATH, f"{vid}.pkl")
    #         with open(vid_path, 'rb') as f:
    #             place = pickle.load(f)
    #         cache[vid] = cache
    #     for sidx in shot_idx:
    #         shot_list.append(place[f"{sidx:04d}"])
    #     shot_list = np.stack(shot_list)
    #     return shot_list

    # def load_other_modality_list(self, vid, shot_idx):
    #     shot_list = []
    #     cache = {}
    #     if vid in cache:
    #         place = cache[vid]
    #     else:
    #         vid_path = os.path.join(self.cfg.OTHER_MODALITY.PATH, f"{vid}.pkl")
    #         with open(vid_path, 'rb') as f:
    #             place = pickle.load(f)
    #         cache[vid] = cache
    #     for sidx in shot_idx:
    #         key = f"{sidx:04d}"
    #         if key in place:
    #             shot = place[key]
    #         else:
    #             shot = np.zeros((257, 90))
    #         shot_list.append(shot)
    #     shot_list = np.stack(shot_list)
    #     return shot_list

    def load_shot(self, vid, sid):
        """
        Args:
            vid: video id
            sid: shot id
        Returns:
            video: list of PIL key-frames
            n_sid: number of key-frames in the shot
        """
        # sid = [int(sid)]
        # video = self.load_shot_list(vid, sid)
        # return video, len(sid)
        sid = [int(sid)]
        video, place, audio = self.load_shot_list(vid, sid)
        if 'PLACE' in self.cfg.OTHER_MODALITY.TYPE:
            place = place[0]
        if 'AUDIO' in self.cfg.OTHER_MODALITY.TYPE:
            audio = audio[0]
        return video, place, audio, len(sid)

    # def load_other_modality(self, vid, sid):
    #     """
    #     Args:
    #         vid: video id
    #         sid: shot id
    #     Returns:
    #         video: list of PIL key-frames
    #         n_sid: number of key-frames in the shot
    #     """
    #     sid = [int(sid)]
    #     video = self.load_other_modality_list(vid, sid)[0]
    #     return video

    def init_transform(self, cfg):
        if self.mode == "extract_shot":
            self.transform = get_transform(cfg.TEST.TRANSFORM)
        elif self.mode == "inference":
            self.transform = get_transform(cfg.TEST.TRANSFORM)
        elif self.mode in ["pretrain", "finetune"]:
            if self.is_train:
                self.transform = get_transform(cfg.TRAIN.TRANSFORM)
            else:
                self.transform = get_transform(cfg.TEST.TRANSFORM)

    def apply_transform(self, images):
        x = torch.stack(self.transform(images), dim=0)
        return x  # [T,3,224,224]

    def init_sampler(self, cfg):
        # shot sampler
        self.shot_sampler = None
        self.sampling_method = cfg.LOSS.sampling_method.name
        logging.info(f"sampling method: {self.sampling_method}")
        sampler_args = copy.deepcopy(
            cfg.LOSS.sampling_method.params.get(self.sampling_method, {})
        )
        print(f"sampling_method: {self.sampling_method}")
        if self.sampling_method == "instance":
            self.shot_sampler = sampler.InstanceShotSampler()
        elif self.sampling_method == "temporal":
            self.shot_sampler = sampler.TemporalShotSampler(**sampler_args)
        elif self.sampling_method == "shotcol":
            self.shot_sampler = sampler.SequenceShotSampler(**sampler_args)
        elif self.sampling_method == "bassl":
            self.shot_sampler = sampler.SequenceShotSampler(**sampler_args)
        elif self.sampling_method == "bassl+shotcol":
            self.shot_sampler = sampler.SequenceShotSampler(**sampler_args)
        elif self.sampling_method == "sbd":
            self.shot_sampler = sampler.NeighborShotSampler(**sampler_args)
        else:
            raise NotImplementedError

    def __repr__(self):
        _repr = "\n"
        _repr += "=" * 8 + "Dataset" + "=" * 8 + "\n"
        _repr += f"Number of Data: {len(self)} \n"
        _repr += "=" * 8 + "=" * 7 + "=" * 8 + "\n"
        return _repr
