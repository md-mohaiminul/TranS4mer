#!/usr/bin/env python3
# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging

from finetune.utils.hydra_utils import print_cfg
from finetune.utils.main_utils import (
    apply_random_seed,
    init_data_loader,
    init_hydra_config,
    init_model,
    init_trainer,
)

from dataset import get_dataset

def main():
    # init cfg
    cfg = init_hydra_config(mode="finetune")
    apply_random_seed(cfg)

    # dataset = get_dataset(cfg, mode="finetune", is_train=True)
    # print('len dataset', len(dataset))

    # item = dataset.__getitem__(100)
    # print(item["video"].shape)
    # print(item["place"].shape)
    # print(item["audio"].shape)

    # init dataloader
    loaders = []
    cfg, train_loader = init_data_loader(cfg, mode="finetune", is_train=True)
    loaders.append(train_loader)
    _, val_loader = init_data_loader(cfg, mode="finetune", is_train=False)
    loaders.append(val_loader)

    # batch = next(iter(train_loader))
    # print(batch['video'].shape)
    #
    # batch = next(iter(val_loader))
    # print(batch['video'].shape)

    #init model
    cfg, model = init_model(cfg)

    # print('#################################################')
    # print(model.loss)

    # init trainer
    # cfg, trainer = init_trainer(cfg)
    #
    # # print cfg
    # print_cfg(cfg)
    #
    # # train
    # logging.info(
    #     f"Start Training: Total Epoch - {cfg.TRAINER.max_epochs}, Precision: {cfg.TRAINER.precision}"
    # )
    # trainer.fit(model, *loaders)
    #trainer.test(model, dataloaders = val_loader)

if __name__ == "__main__":
    main()
