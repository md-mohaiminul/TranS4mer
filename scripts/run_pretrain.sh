#!/usr/bin/env bash
# script for pre-training BaSSL

#LOAD_FROM=bassl_vit_small

#PLACE_PATH=/playpen-storage/mmiemon/datasets/movienet/place_feat_1K

method=bassl
EXPR_NAME=test
WORK_DIR=$(pwd) # == <path_to_root>/bassl
PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/pretrain/main.py \
	config.EXPR_NAME=${EXPR_NAME} \
	config.TRAIN.BATCH_SIZE.effective_batch_size=512 \
	config.TRAIN.NUM_WORKERS=16 \
	config.DISTRIBUTED.NUM_NODES=1 \
	config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
	+method=${method} \
