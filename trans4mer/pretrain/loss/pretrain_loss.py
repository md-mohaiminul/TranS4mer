# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging

import numpy as np
import torch
import torch.nn.functional as F
from pretrain.loss.pretext_task import PretextTaskWrapper, SimclrLoss


class InstanceSimclrLoss(SimclrLoss):
    def __init__(self, cfg):
        SimclrLoss.__init__(self, cfg=cfg, is_bassl=False)


class TemporalSimclrLoss(SimclrLoss):
    def __init__(self, cfg):
        SimclrLoss.__init__(self, cfg=cfg, is_bassl=False)


class ShotColSimclrLoss(SimclrLoss):
    def __init__(self, cfg):
        SimclrLoss.__init__(self, cfg=cfg, is_bassl=False)

        # to disable debug dump in numba (used by DTW computation)
        numba_logger = logging.getLogger("numba")
        numba_logger.setLevel(logging.WARNING)

    def forward(self, shot_repr, **kwargs):
        # shot_repr shape: [b t d] where t = neighbor_size * 2 + 1
        b, t, d = shot_repr.shape
        n_sparse = kwargs.get("n_sparse", -1)
        n_dense = kwargs.get("n_dense", -1)

        # infer head to obtain embedding
        # sparse sequence includes three shots [first, last, center] from dense one
        # but, diffrent augmentation applied
        head_shot_repr = self.head_nce(shot_repr)
        s_emb, d_emb = torch.split(head_shot_repr, [n_sparse, n_dense], dim=1)

        # get NN shot index
        with torch.no_grad():
            center_idx = n_dense // 2
            normalized_d_emb = F.normalize(d_emb, dim=2)
            sim = torch.einsum(
                "bd,btd->bt", normalized_d_emb[:, center_idx], normalized_d_emb
            )
            sim[:, center_idx] = -10000.0
            nn_idx = torch.argmax(sim, dim=1)

        # compute simclr loss between center shot (of sparse) and NN shot (of dense)
        b_idx = torch.arange(0, b, device=shot_repr.device)
        pos_neg_emb = torch.cat(
            [s_emb[:, -1], d_emb[b_idx, nn_idx]], dim=0
        )  # [(2 b) d]
        loss = {"simclr_loss": self._compute_nce_loss(pos_neg_emb)}

        return loss


class BaSSLLoss(PretextTaskWrapper):
    def __init__(self, cfg):
        PretextTaskWrapper.__init__(self, cfg=cfg)

        # to disable debug dump in numba (used by DTW computation)
        numba_logger = logging.getLogger("numba")
        numba_logger.setLevel(logging.WARNING)

        self.use_ssm_loss = cfg.LOSS.shot_scene_matching.get("enabled", True)
        self.it = 0

    def forward(self, shot_repr, **kwargs):
        self.it += 1
        b, t, d = shot_repr.shape
        n_sparse = kwargs.get("n_sparse", -1)
        n_dense = kwargs.get("n_dense", -1)

        sparse_idx = kwargs.get("sparse_idx", -1)  #added later

        # sim_trans
        # sim_trans = kwargs.get("sim_trans", None)
        # shot_repr = sim_trans(shot_repr, place, audio)

        # obtain embeddings using NCE head from sparse and dense shot sequences
        # head_shot_repr: [b n_sparse+n_dense d] -> [b n_sparse d], [b n_dense d]

        # shot_repr[:,0,:] = shot_repr[:,2,:]
        # shot_repr[:,1,:] = shot_repr[:,18,:]

        head_shot_repr = self.head_nce(shot_repr)
        s_emb, d_emb = torch.split(head_shot_repr, [n_sparse, n_dense], dim=1)

        #added later
        # d_emb = head_shot_repr
        # indices = [0,16]
        # s_emb = head_shot_repr[:,indices,:]
        # dense_shot_repr = shot_repr

        # head_place_repr = self.head_nce_place(place)
        # s_place, d_place = torch.split(head_place_repr, [n_sparse, n_dense], dim=1)
        # s_emb_w_place = torch.cat([s_emb, s_place], dim=-1)
        # d_emb_w_place = torch.cat([d_emb, d_place], dim=-1)

        #print(place.shape, head_place_repr.shape, s_emb_w_place.shape, d_emb_w_place.shape)

        # _, dense_image = torch.split(shot_repr, [n_sparse, n_dense], dim=1)
        # _, dense_place = torch.split(place, [n_sparse, n_dense], dim=1)
        # dense_image = F.normalize(dense_image)
        # dense_place = F.normalize(dense_place)
        # sim1 = torch.sum(dense_image[:, :-1, :] * dense_image[:, 1:, :], dim=-1)
        # sim2 = torch.sum(dense_place[:, :-1, :] * dense_place[:, 1:, :], dim=-1)

        # sim = torch.sum(d_emb[:, :-1, :] * d_emb[:, 1:, :], dim=-1)
        # boundary = torch.argmin(sim, dim=-1)

        # compute alignment between sparse and dense sequences using DTW
        dtw_path = self._compute_dtw_path(s_emb, d_emb)

        # dtw_path = self._compute_dtw_path(s_emb_w_place, d_emb_w_place)
        # for b in range(len(dtw_path)):
        #     if dtw_path[b][-2][0]==0:
        #         dtw_path[b][-2][0] = 1

        # for i in range(d_emb.shape[0]):
        #     x = np.zeros(d_emb.shape[1])
        #     x[boundary[i]+1:] = 1
        #     dtw_path[i][:, 0] = x

        loss = {}
        if self.use_crn:
            masking_mask = kwargs.get("mask", None)
            crn = kwargs.get("crn", None)
            assert crn is not None

            # obtain sparse and dense shot_repr: [b n_sparse d], [b n_dense d]
            _, dense_shot_repr = torch.split(shot_repr, [n_sparse, n_dense], dim=1)

        # compute masked shot modeling loss
        if self.use_msm_loss:
            masked_shot_loss = self._compute_msm_loss(
                crn, dense_shot_repr, masking_mask
            )
            loss["msm_loss"] = masked_shot_loss

        # compute shot-scene matching Loss
        if self.use_ssm_loss:
            ssm_loss = self._compute_ssm_loss(s_emb, d_emb, dtw_path)
            loss["ssm_loss"] = ssm_loss

        if self.use_pp_loss or self.use_cgm_loss:
            crn_repr_wo_mask, _ = crn(dense_shot_repr)  # infer CRN without masking
            crn_repr_wo_mask = crn_repr_wo_mask[
                :, 1:
            ].contiguous()  # exclude [CLS] token

            #crn_repr_wo_mask = dense_shot_repr

            # obtain offset (index) of boundary shot
            bd_idx = self._compute_boundary(dtw_path, n_dense)

        # compute pseudo-boundary prediction loss
        if self.use_pp_loss:
            pp_loss = self._compute_pp_loss(crn_repr_wo_mask, bd_idx)
            loss["pp_loss"] = pp_loss

        # compute contextual group matching loss
        if self.use_cgm_loss:
            cgm_loss = self._compute_cgm_loss(crn_repr_wo_mask, dtw_path, bd_idx)
            loss["cgm_loss"] = cgm_loss

        return loss
        #return bd_idx


class BaSSLShotcolSimclrLoss(PretextTaskWrapper):
    def __init__(self, cfg):
        PretextTaskWrapper.__init__(self, cfg=cfg)

        # to disable debug dump in numba (used by DTW computation)
        numba_logger = logging.getLogger("numba")
        numba_logger.setLevel(logging.WARNING)

        self.use_contrastive_loss = cfg.LOSS.shot_scene_matching.get("enabled", True)

    def forward(self, shot_repr, **kwargs):
        b, t, d = shot_repr.shape
        n_sparse = kwargs.get("n_sparse", -1)
        n_dense = kwargs.get("n_dense", -1)

        # obtain embeddings of sparse and dense shot for head network
        # head_shot_repr: [b n_sparse+n_dense d] -> [b n_sparse d], [b n_dense d]
        head_shot_repr = self.head_nce(shot_repr)
        _s_emb, d_emb = torch.split(head_shot_repr, [n_sparse, n_dense], dim=1)
        s_emb = _s_emb[:, :2]  # (first, last)
        center_s_emb = _s_emb[:, -1]  # center

        # compute alignment between sparse and dense sequences using DTW
        dtw_path = self._compute_dtw_path(s_emb, d_emb)

        loss = {}
        if self.use_crn:
            masking_mask = kwargs.get("mask", None)
            crn = kwargs.get("crn", None)
            assert crn is not None

            # obtain sparse and dense shot shot_repr: [b n_sparse d], [b n_dense d]
            _, dense_shot_repr = torch.split(shot_repr, [n_sparse, n_dense], dim=1)

        # compute masked shot modeling loss
        if self.use_msm_loss:
            masked_shot_loss = self._compute_msm_loss(
                crn, dense_shot_repr, masking_mask
            )
            loss["msm_loss"] = masked_shot_loss

        if self.use_contrastive_loss:
            # get NN shot index
            with torch.no_grad():
                center_idx = n_dense // 2
                normalized_d_emb = F.normalize(d_emb, dim=2)
                sim = torch.einsum(
                    "bd,btd->bt", normalized_d_emb[:, center_idx], normalized_d_emb
                )
                sim[:, center_idx] = -10000.0
                nn_idx = torch.argmax(sim, dim=1)

            # compute shotcol loss between center shot (of sparse) and NN shot (of dense)
            b_idx = torch.arange(0, b, device=shot_repr.device)
            pos_neg_emb = torch.cat(
                [center_s_emb, d_emb[b_idx, nn_idx]], dim=0
            )  # [(2 b) d]
            loss["shotcol"] = self._compute_nce_loss(pos_neg_emb)

            # compute shot-scene matching Loss
            ssm_loss = self._compute_ssm_loss(s_emb, d_emb, dtw_path)
            loss["ssm_loss"] = ssm_loss

        if self.use_pp_loss or self.use_cgm_loss:
            crn_repr_wo_mask, _ = crn(dense_shot_repr)  # infer CRN without masking
            crn_repr_wo_mask = crn_repr_wo_mask[
                :, 1:
            ].contiguous()  # exclude [CLS] token

            # obtain offset (index) of boundary shot
            bd_idx = self._compute_boundary(dtw_path, n_dense)

        # compute pseudo-boundary prediction loss
        if self.use_pp_loss:
            pp_loss = self._compute_pp_loss(crn_repr_wo_mask, bd_idx)
            loss["pp_loss"] = pp_loss

        # compute contextual group matching loss
        if self.use_cgm_loss:
            cgm_loss = self._compute_cgm_loss(crn_repr_wo_mask, dtw_path, bd_idx)
            loss["cgm_loss"] = cgm_loss

        return loss