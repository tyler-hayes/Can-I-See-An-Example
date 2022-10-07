#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Can-I-See-An-Example
export PYTHONPATH=${PROJ_ROOT}
source activate long_tail_continual_active_learning
cd ${PROJ_ROOT}/train_continual_active_learners

BASE_DIR=/media/tyler/Data/codes/Long-Tail-Active-Learning/results/

# TRAIN OFFLINE MODELS
for NETWORK_SEED in 0 1 2 3 4 5 6 7 8 9; do
  EXPT_NAME=triple_completion_full_dataset_baseline_sgd_network_seed_${NETWORK_SEED}
  echo "Experiment: ${EXPT_NAME}"
  OUTPUT_DIR=${BASE_DIR}${EXPT_NAME}

  python -u train_triple_completion_qtype.py \
    --output_dir ${OUTPUT_DIR} \
    --optimizer sgd \
    --pre_training 1 \
    --ckpt_epoch 5 \
    --network_seed ${NETWORK_SEED} \
    --num_attribute_categories 253 \
    --num_predicate_categories 46 \
    --epochs 25 >${BASE_DIR}logs/${EXPT_NAME}.log
done

# EVALUATE OFFLINE MODELS ON ALL TEST SETS
for NETWORK_SEED in 0 1 2 3 4 5 6 7 8 9; do
  EXPT_NAME=triple_completion_full_dataset_baseline_sgd_network_seed_${NETWORK_SEED}
  echo "Evaluating: ${EXPT_NAME}"
  OUTPUT_DIR=${BASE_DIR}${EXPT_NAME}
  CKPT_FILE=${OUTPUT_DIR}/triple_completion_model_final_ckpt.pth

  python -u train_triple_completion_qtype.py \
    --output_dir ${OUTPUT_DIR} \
    --ckpt_file ${CKPT_FILE} \
    --optimizer sgd \
    --pre_training 1 \
    --ckpt_epoch 5 \
    --network_seed ${NETWORK_SEED} \
    --num_attribute_categories 253 \
    --num_predicate_categories 46 \
    --epochs 25 \
    --eval_only 1 \
    --test_tail 1 >${BASE_DIR}logs/${EXPT_NAME}_eval.log
done
