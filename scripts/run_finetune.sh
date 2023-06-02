#!/usr/bin/env bash
# script for fine-tuning BaSSL

#LOAD_FROM=bassl_imagenet
LOAD_FROM=bassl_vit_small_gs4_factorized
WORK_DIR=$(pwd)

#extract shot representation
#PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/pretrain/extract_shot_repr.py \
#		config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
#	    +config.LOAD_FROM=${LOAD_FROM}
#sleep 100s

# finetune the model
EXPR_NAME=test
PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/finetune/main.py \
	config.TRAIN.BATCH_SIZE.effective_batch_size=1024 \
	config.TRAIN.NUM_WORKERS=16 \
	config.DISTRIBUTED.NUM_NODES=1 \
	config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
	config.EXPR_NAME=${EXPR_NAME} \
	+config.PRETRAINED_LOAD_FROM=${LOAD_FROM}
