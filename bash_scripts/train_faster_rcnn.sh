#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Can-I-See-An-Example
export PYTHONPATH=${PROJ_ROOT}
source activate long_tail_continual_active_learning
cd ${PROJ_ROOT}/train_object_detection_feature_extractor

RESULTS_DIR=/media/tyler/Data/codes/Long-Tail-Active-Learning/results/
COCO_IMAGE_DIR=/media/tyler/Data/datasets/COCO/

EXPT_NAME=faster_rcnn_train
echo "Experiment: ${EXPT_NAME}"
OUTPUT_DIR=${RESULTS_DIR}/results/${EXPT_NAME}

# TRAIN FASTER-RCNN + RESNET-50 BACKBONE ON COCO FOR FEATURE EXTRACTION
python -u train.py \
  --data-path ${COCO_IMAGE_DIR} \
  --output-dir ${OUTPUT_DIR} >${RESULTS_DIR}logs/${EXPT_NAME}.log
