#!/usr/bin/env bash
# script for fine-tuning BaSSL

#EXPR_NAME=bassl_vit_small
#EXPR_NAME=bassl_vit_s4
EXPR_NAME=bassl_vit_small_gs4
#EXPR_NAME=bassl_imagenet
LOAD_FROM=finetune/ckpt/${EXPR_NAME}/model-v1.ckpt
WORK_DIR=$(pwd)

PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/finetune/inference.py \
	config.TRAIN.BATCH_SIZE.effective_batch_size=1024 \
	config.TRAIN.NUM_WORKERS=16 \
	config.DISTRIBUTED.NUM_NODES=1 \
	config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
	config.EXPR_NAME=${EXPR_NAME} \
	+config.PRETRAINED_LOAD_FROM=${EXPR_NAME} \
	+config.FINETUNED_LOAD_FROM=${LOAD_FROM}
