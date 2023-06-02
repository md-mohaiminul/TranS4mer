# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import json
import logging
import os

import einops
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from finetune.utils.hydra_utils import save_config_to_disk
from finetune.utils.metric import (
    AccuracyMetric,
    F1ScoreMetric,
    MovieNetMetric,
    SklearnAPMetric,
    SklearnAUCROCMetric,
)
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

import numpy as np
from tslearn import metrics

class FinetuningWrapper(pl.LightningModule):
    def __init__(self, cfg, shot_encoder, crn, loss=None):
        super().__init__()
        self.cfg = cfg

        # build model components
        self.shot_encoder = shot_encoder
        self.crn = crn
        #self.sim_trans = sim_trans
        crn_name = cfg.MODEL.contextual_relation_network.name
        hdim = cfg.MODEL.contextual_relation_network.params[crn_name]["hidden_size"]
        self.head_sbd = nn.Linear(hdim, 2)

        # define metrics
        self.acc_metric = AccuracyMetric()
        self.ap_metric = SklearnAPMetric()
        self.f1_metric = F1ScoreMetric(num_classes=1)
        self.auc_metric = SklearnAUCROCMetric()
        self.movienet_metric = MovieNetMetric()

        self.log_dir = os.path.join(cfg.LOG_PATH, cfg.EXPR_NAME)
        self.use_raw_shot = cfg.USE_RAW_SHOT
        self.eps = 1e-5

        #added later
        self.predictions = []
        self.best_ap = 0
        self.all_preds = {}

    def on_train_start(self) -> None:
        if self.global_rank == 0:
            try:
                save_config_to_disk(self.cfg)
            except Exception as err:
                logging.info(err)

    def extract_shot_representation(self, inputs: torch.Tensor) -> torch.Tensor:
        """ inputs [b s k c h w] -> output [b d] """
        assert len(inputs.shape) == 6  # (B Shot Keyframe C H W)
        b, s, k, c, h, w = inputs.shape   #[32, 17, 1, 3, 224, 224]

        # we extract feature of each key-frame and average them
        inputs = einops.rearrange(inputs, "b s k c h w -> (b s) k c h w", s=s)
        keyframe_repr = [self.shot_encoder(inputs[:, _k]) for _k in range(k)]
        shot_repr = torch.stack(keyframe_repr).mean(dim=0)  # [k (b s) d] -> [(b s) d]
        shot_repr = einops.rearrange(shot_repr, "(b s) d -> b s d", s=s)  # [32, 17, 2048]

        return shot_repr

    def shared_step(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # infer shot encoder
            if self.use_raw_shot:
                shot_repr = self.extract_shot_representation(inputs)
            else:
                shot_repr = inputs
        assert len(shot_repr.shape) == 3

        # infer CRN
        if self.cfg.MODEL.contextual_relation_network.enabled:
            _, pooled = self.crn(shot_repr, mask=None)
        else:
            cidx = shot_repr.shape[1] // 2
            pooled = shot_repr[:, cidx, :]
        # infer boundary score
        # if self.loss:
        #     pred = self.loss.head_pp(pooled)
        # else:
        pred = self.head_sbd(pooled)
        return pred

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.shared_step(x, **kwargs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs = batch["video"]
        labels = batch["label"]
        outputs = self.shared_step(inputs)

        # compute sbd loss where positive and negative ones are
        # balanced with their numbers
        loss = F.cross_entropy(outputs.squeeze(), labels.squeeze(), reduction="none")
        lpos = labels == 1
        lneg = labels == 0

        pp, nn = 1, 1
        #pp, nn = 4, 1
        wp = (pp / float(pp + nn)) * lpos / (lpos.sum() + self.eps)
        wn = (nn / float(pp + nn)) * lneg / (lneg.sum() + self.eps)
        w = wp + wn
        loss = (w * loss).sum()

        # write metrics
        preds = torch.argmax(outputs, dim=1)

        gt_one = labels == 1
        gt_zero = labels == 0
        pred_one = preds == 1
        pred_zero = preds == 0

        tp = (gt_one * pred_one).sum()
        fp = (gt_zero * pred_one).sum()
        tn = (gt_zero * pred_zero).sum()
        fn = (gt_one * pred_zero).sum()

        acc0 = 100.0 * tn / (fp + tn + self.eps)
        acc1 = 100.0 * tp / (tp + fn + self.eps)
        tp_tn = tp + tn

        self.log(
            "sbd_train/loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/tp_batch",
            tp,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/fp_batch",
            fp,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/tn_batch",
            tn,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/fn_batch",
            fn,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/acc0",
            acc0,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/acc1",
            acc1,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/tp_tn",
            tp_tn,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    #added later
    @torch.no_grad()
    def _compute_dtw_path(self, s_emb, d_emb):
        """ compute alignment between two sequences using DTW """
        cost = (
            (1 - torch.bmm(s_emb, d_emb.transpose(1, 2)))
                .cpu()
                .numpy()
                .astype(np.float32)
        )  # shape: [b n_sparse n_dense]
        dtw_path = []
        for bi in range(cost.shape[0]):
            _path, _ = metrics.dtw_path_from_metric(cost[bi], metric="precomputed")
            dtw_path.append(np.asarray(_path))  # [n_dense 2]

        return dtw_path

    def _compute_boundary(self, dtw_path, nshot):
        """ get indices of boundary shots
        return:
            bd_idx: list of size B each of which means index of boundary shot
        """
        # dtw_path: list of B * [ndense 2]
        # find boundary location where the last index of first group (0)
        np_path = np.asarray(dtw_path)
        bd_idx = [np.where(path[:, 0] == 0)[0][-1] for path in np_path]

        return bd_idx

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        vids = batch["vid"]
        sids = batch["sid"]
        inputs = batch["video"]  #[1024, 17, 2048]
        labels = batch["label"]  #[1024]
        outputs = self.shared_step(inputs) #[1024, 2]

        prob = F.softmax(outputs, dim=1)   #[1024, 2]
        preds = torch.argmax(prob, dim=1)  #[1024]

        #print(inputs.shape, outputs.shape, labels.shape, prob.shape, preds.shape)

        # indices = torch.tensor([0, 16]).cuda()
        # sparse = torch.index_select(inputs, 1, indices)   #[1024, 2, 2048]
        # print(sparse.shape)

        self.acc_metric.update(
            prob[:, 1], labels
        )  # prob[:,1] is confidence score for boundary
        self.ap_metric.update(prob[:, 1], labels)
        self.f1_metric.update(prob[:, 1], labels)
        self.auc_metric.update(prob[:, 1], labels)

        #removed for BBC
        # for vid, sid, pred, gt in zip(vids, sids, preds, labels):
        #     self.movienet_metric.update(vid, sid, pred, gt)
        #
        #     #added later
        #     self.predictions.append([vid, sid, gt.item(), pred.item()])


    def validation_epoch_end(self, validation_step_outputs):
        score = {}

        # update acc.
        acc = self.acc_metric.compute()
        torch.cuda.synchronize()
        assert isinstance(acc, dict)
        score.update(acc)

        # update average precision (AP).
        ap, _, _ = self.ap_metric.compute()  # * 100.
        ap *= 100.0
        torch.cuda.synchronize()
        assert isinstance(ap, torch.Tensor)
        score.update({"ap": ap})

        # update AUC-ROC
        auc, _, _ = self.auc_metric.compute()
        auc *= 100.0
        torch.cuda.synchronize()
        assert isinstance(auc, torch.Tensor)
        score.update({"auc": auc})

        # update F1 score.
        f1 = self.f1_metric.compute() * 100.0
        torch.cuda.synchronize()
        assert isinstance(f1, torch.Tensor)
        score.update({"f1": f1})

        # update recall, mIoU score.
        # removed for BBC
        # recall, recall_at_3s, miou = self.movienet_metric.compute()
        # torch.cuda.synchronize()
        # assert isinstance(recall, torch.Tensor)
        # assert isinstance(recall_at_3s, torch.Tensor)
        # assert isinstance(miou, torch.Tensor)
        # score.update({"recall": recall * 100.0})
        # score.update({"recall@3s": recall_at_3s * 100})
        # score.update({"mIoU": miou * 100})

        # logging
        for k, v in score.items():
            self.log(
                f"sbd_test/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        score = {k: v.item() for k, v in score.items()}
        self.print(f"\nTest Score: {score}")

        # reset all metrics
        self.acc_metric.reset()
        self.ap_metric.reset()
        self.f1_metric.reset()
        self.auc_metric.reset()
        self.movienet_metric.reset()

        if score['ap']>self.best_ap:
            self.best_ap = score['ap']
            # save last epoch result.
            with open(os.path.join(self.log_dir, "all_score.json"), "w") as fopen:
                json.dump(score, fopen, indent=4, ensure_ascii=False)

            #added later save predictions
            with open(os.path.join(self.log_dir, "predictions.json"), "w") as fopen:
                json.dump(self.predictions, fopen, indent=4, ensure_ascii=False)
        self.predictions.clear()

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self.validation_step(batch, batch_idx)

        #comcast
        # vids = batch["vid"]
        # sids = batch["sid"]
        # inputs = batch["video"]  # [1024, 17, 2048]
        # labels = batch["label"]  # [1024]
        # outputs = self.shared_step(inputs)  # [1024, 2]
        #
        # prob = F.softmax(outputs, dim=1)  # [1024, 2]
        # preds = torch.argmax(prob, dim=1)  # [1024]
        #
        # for vid, pred in zip(vids, preds):
        #     self.all_preds[vid] = pred.item()

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)
        #comcast
        # with open(f'outputs/{self.cfg.EXPR_NAME}.txt', 'w') as convert_file:
        #     convert_file.write(json.dumps(self.all_preds))
        # print("test end")

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        # params
        skip_list = []
        weight_decay = self.cfg.TRAIN.OPTIMIZER.weight_decay
        if not self.cfg.TRAIN.OPTIMIZER.regularize_bn:
            skip_list.append("bn")
        if not self.cfg.TRAIN.OPTIMIZER.regularize_bias:
            skip_list.append("bias")
        params = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=weight_decay, skip_list=skip_list
        )

        # optimizer
        if self.cfg.TRAIN.OPTIMIZER.name == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.cfg.TRAIN.OPTIMIZER.lr.scaled_lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif self.cfg.TRAIN.OPTIMIZER.name == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.cfg.TRAIN.OPTIMIZER.lr.scaled_lr
            )
        else:
            raise ValueError()

        warmup_steps = int(
            self.cfg.TRAIN.TRAIN_ITERS_PER_EPOCH
            * self.cfg.TRAINER.max_epochs
            * self.cfg.TRAIN.OPTIMIZER.scheduler.warmup
        )
        total_steps = int(
            self.cfg.TRAIN.TRAIN_ITERS_PER_EPOCH * self.cfg.TRAINER.max_epochs
        )

        if self.cfg.TRAIN.OPTIMIZER.scheduler.name == "cosine_with_linear_warmup":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]
