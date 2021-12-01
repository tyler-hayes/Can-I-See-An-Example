#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Continual-Active-Learn-Scene-Graphs
export PYTHONPATH=${PROJ_ROOT}
source activate long_tail_continual_active_learning
cd ${PROJ_ROOT}/train_continual_active_learners

BASE_DIR=/media/tyler/Data/codes/Long-Tail-Active-Learning/results/

PERMUTATION_SEED=444
NUM_PRE_TRAIN_SAMPLES=2500

# TRAIN PRE-TRAIN MODELS
for NETWORK_SEED in 0 1 2 3 4 5 6 7 8 9; do
  EXPT_NAME=triple_completion_baseline_sgd_num_head_iter_samples_${NUM_PRE_TRAIN_SAMPLES}_network_seed_${NETWORK_SEED}_permutation_seed_${PERMUTATION_SEED}
  echo "Experiment: ${EXPT_NAME}"
  OUTPUT_DIR=${BASE_DIR}/${EXPT_NAME}

  python -u train_triple_completion_qtype.py \
    --output_dir ${OUTPUT_DIR} \
    --pre_training 1 \
    --pre_train_iter_level 1 \
    --optimizer sgd \
    --network_seed ${NETWORK_SEED} \
    --class_permutation_seed ${PERMUTATION_SEED} \
    --ckpt_epoch 1000 \
    --num_attribute_categories 253 \
    --num_predicate_categories 46 \
    --epochs 1 \
    --num_head_samples_per_class ${NUM_PRE_TRAIN_SAMPLES} >${BASE_DIR}logs/${EXPT_NAME}.log
done

# EVALUATE PRE-TRAIN MODELS ON ALL TEST SETS
for NETWORK_SEED in 0 1 2 3 4 5 6 7 8 9; do
  EXPT_NAME=triple_completion_baseline_sgd_num_head_iter_samples_${NUM_PRE_TRAIN_SAMPLES}_network_seed_${NETWORK_SEED}_permutation_seed_${PERMUTATION_SEED}
  echo "Evaluating: ${EXPT_NAME}"
  OUTPUT_DIR=${BASE_DIR}${EXPT_NAME}
  CKPT_FILE=${OUTPUT_DIR}/triple_completion_model_final_ckpt.pth

  python -u train_triple_completion_qtype.py \
    --output_dir ${OUTPUT_DIR} \
    --ckpt_file ${CKPT_FILE} \
    --optimizer sgd \
    --pre_training 1 \
    --ckpt_epoch 1000 \
    --network_seed ${NETWORK_SEED} \
    --num_attribute_categories 253 \
    --num_predicate_categories 46 \
    --epochs 1 \
    --eval_only 1 \
    --test_tail 1 >${BASE_DIR}logs/${EXPT_NAME}_eval.log
done
