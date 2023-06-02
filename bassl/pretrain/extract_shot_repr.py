#!/usr/bin/env python3
# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging
import os

from pretrain.utils.hydra_utils import print_cfg
from pretrain.utils.main_utils import (
    apply_random_seed,
    init_data_loader,
    init_hydra_config,
    init_model,
    init_trainer,
    load_pretrained_config,
)


def main():
    # init cfg
    cfg = init_hydra_config(mode="extract_shot")
    apply_random_seed(cfg)
    cfg = load_pretrained_config(cfg)

    # cfg.TRAIN.USE_DOUBLE_KEYFRAME = False
    #ViT
    # cfg.MODEL.shot_encoder.name = 'vit'
    # cfg.MODEL.shot_encoder.vit.attention_type = 'gs4'
    # cfg.MODEL.contextual_relation_network.params.trn.input_dim = 384
    # cfg.LOSS.shot_scene_matching.params.simclr_loss.head.input_dim = 384
    # cfg.LOSS.shot_scene_matching.params.simclr_loss.head.hidden_dim = 768

    # cfg.TRAIN.BATCH_SIZE.effective_batch_size = 512
    # cfg.OTHER_MODALITY.TYPE = []

    #cfg.DISTRIBUTED.NUM_PROC_PER_NODE = 8

    #BBC
    # cfg.DATASET = 'BBC'
    # cfg.DATA_PATH = "./data/BBC"
    # cfg.IMG_PATH = os.path.join(cfg.DATA_PATH, "frames")
    # cfg.ANNO_PATH = os.path.join(cfg.DATA_PATH, "anno")
    # cfg.FEAT_PATH = os.path.join(cfg.DATA_PATH, "features")

    print_cfg(cfg)

    # init dataloader
    cfg, test_loader = init_data_loader(cfg, mode="extract_shot", is_train=False)

    # init model
    cfg, model = init_model(cfg)

    # init trainer
    cfg, trainer = init_trainer(cfg)

    batch = next(iter(test_loader))
    print(batch['video'].shape)

    # train
    logging.info(f"Start Inference: {cfg.LOAD_FROM}")
    #trainer.test(model, test_dataloaders=test_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
