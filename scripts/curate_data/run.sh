#!/bin/bash
HF_USER=NONHUMAN-RESEARCH
TASK_NAME="VENDA"
REPO_NAME=SOARM100_TASK_${TASK_NAME}
OUTPUT_DIR=outputs/train/${REPO_NAME}_V2
JOB_NAME=${REPO_NAME}

python train_vaes.py \
  --dataset.repo_id=${HF_USER}/${REPO_NAME} \
  --policy.type=act \
  --output_dir=${OUTPUT_DIR} \
  --batch_size=8 \
  --type=actions