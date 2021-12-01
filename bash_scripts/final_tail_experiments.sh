#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Continual-Active-Learn-Scene-Graphs
export PYTHONPATH=${PROJ_ROOT}
source activate long_tail_continual_active_learning
cd ${PROJ_ROOT}/train_continual_active_learners

BASE_DIR=/media/tyler/Data/codes/Long-Tail-Active-Learning/results/
PRE_TRAIN_DIR=${BASE_DIR}pre-train/seed

NUM_INC=10
SIZE=600
LR=0.01
PERMUTATION_SEED=444
NUM_PRE_TRAIN_SAMPLES=2500
EPOCHS=100
VAL_EPOCH=5
PATIENCE=10
METHOD=tail

HARD_NEGATIVE_TYPE=reservoir_hard

# TAIL RE-BALANCED MINI-BATCHES EXPERIMENTS
for NETWORK_SEED in 0 1 2 3 4 5 6 7 8 9; do

  # SAVED OUT PRE-TRAIN CKPTS
  EXPT_NAME_PRE_TRAIN=triple_completion_baseline_sgd_num_head_iter_samples_${NUM_PRE_TRAIN_SAMPLES}_network_seed_${NETWORK_SEED}_permutation_seed_${PERMUTATION_SEED}
  CKPT=${PRE_TRAIN_DIR}${NETWORK_SEED}/${EXPT_NAME_PRE_TRAIN}/triple_completion_model_final_ckpt.pth

  EXPT_NAME=final_active_learning_head_pre_train_bal_qtype_method_${METHOD}_rank_equal_proba_old_new_bias_correction_${SIZE}_num_head_samples_${NUM_PRE_TRAIN_SAMPLES}_network_seed_${NETWORK_SEED}_permutation_seed_${PERMUTATION_SEED}
  echo "Experiment: ${EXPT_NAME}"
  OUTPUT_DIR=${BASE_DIR}${EXPT_NAME}

  python -u train_triple_completion_qtype.py \
    --output_dir ${OUTPUT_DIR} \
    --sampling_type balanced_rank \
    --tail_seen_distribution 0 \
    --test_batch_size 512 \
    --use_bias_correction 1 \
    --cross_validate_selection 1 \
    --cross_validation_patience ${PATIENCE} \
    --hard_negatives_type ${HARD_NEGATIVE_TYPE} \
    --epochs ${EPOCHS} \
    --val_epoch ${VAL_EPOCH} \
    --optimizer sgd \
    --lr ${LR} \
    --active_learning_method ${METHOD} \
    --num_active_learning_samples ${SIZE} \
    --num_active_learning_increments ${NUM_INC} \
    --pre_training 0 \
    --network_seed ${NETWORK_SEED} \
    --class_permutation_seed ${PERMUTATION_SEED} \
    --num_attribute_categories 253 \
    --num_predicate_categories 46 \
    --num_head_samples_per_class ${NUM_PRE_TRAIN_SAMPLES} \
    --ckpt_file ${CKPT} >${BASE_DIR}logs/${EXPT_NAME}.log
done

# TAIL STANDARD MINI-BATCHES EXPERIMENTS
for NETWORK_SEED in 0 1 2 3 4 5 6 7 8 9; do

  # SAVED OUT PRE-TRAIN CKPTS
  EXPT_NAME_PRE_TRAIN=triple_completion_baseline_sgd_num_head_iter_samples_${NUM_PRE_TRAIN_SAMPLES}_network_seed_${NETWORK_SEED}_permutation_seed_${PERMUTATION_SEED}
  CKPT=${PRE_TRAIN_DIR}${NETWORK_SEED}/${EXPT_NAME_PRE_TRAIN}/triple_completion_model_final_ckpt.pth

  EXPT_NAME=final_active_learning_head_pre_train_bal_qtype_method_${METHOD}_rank_equal_proba_standard_mini_batch_${SIZE}_num_head_samples_${NUM_PRE_TRAIN_SAMPLES}_network_seed_${NETWORK_SEED}_permutation_seed_${PERMUTATION_SEED}
  echo "Experiment: ${EXPT_NAME}"
  OUTPUT_DIR=${BASE_DIR}${EXPT_NAME}

  python -u train_triple_completion_qtype.py \
    --output_dir ${OUTPUT_DIR} \
    --sampling_type balanced_rank \
    --use_standard_mini_batches 1 \
    --tail_seen_distribution 0 \
    --test_batch_size 512 \
    --cross_validate_selection 1 \
    --cross_validation_patience ${PATIENCE} \
    --hard_negatives_type ${HARD_NEGATIVE_TYPE} \
    --epochs ${EPOCHS} \
    --val_epoch ${VAL_EPOCH} \
    --optimizer sgd \
    --lr ${LR} \
    --active_learning_method ${METHOD} \
    --num_active_learning_samples ${SIZE} \
    --num_active_learning_increments ${NUM_INC} \
    --pre_training 0 \
    --network_seed ${NETWORK_SEED} \
    --class_permutation_seed ${PERMUTATION_SEED} \
    --num_attribute_categories 253 \
    --num_predicate_categories 46 \
    --num_head_samples_per_class ${NUM_PRE_TRAIN_SAMPLES} \
    --ckpt_file ${CKPT} >${BASE_DIR}logs/${EXPT_NAME}.log
done

# TAIL COUNT PROBABILITY EXPERIMENTS
for NETWORK_SEED in 0 1 2 3 4 5 6 7 8 9; do

  # SAVED OUT PRE-TRAIN CKPTS
  EXPT_NAME_PRE_TRAIN=triple_completion_baseline_sgd_num_head_iter_samples_${NUM_PRE_TRAIN_SAMPLES}_network_seed_${NETWORK_SEED}_permutation_seed_${PERMUTATION_SEED}
  CKPT=${PRE_TRAIN_DIR}${NETWORK_SEED}/${EXPT_NAME_PRE_TRAIN}/triple_completion_model_final_ckpt.pth

  EXPT_NAME=final_active_learning_head_pre_train_bal_qtype_method_${METHOD}_rank_old_new_bias_correction_${SIZE}_num_head_samples_${NUM_PRE_TRAIN_SAMPLES}_network_seed_${NETWORK_SEED}_permutation_seed_${PERMUTATION_SEED}
  echo "Experiment: ${EXPT_NAME}"
  OUTPUT_DIR=${BASE_DIR}${EXPT_NAME}

  python -u train_triple_completion_qtype.py \
    --output_dir ${OUTPUT_DIR} \
    --sampling_type balanced_rank \
    --test_batch_size 512 \
    --use_bias_correction 1 \
    --cross_validate_selection 1 \
    --cross_validation_patience ${PATIENCE} \
    --hard_negatives_type ${HARD_NEGATIVE_TYPE} \
    --epochs ${EPOCHS} \
    --val_epoch ${VAL_EPOCH} \
    --optimizer sgd \
    --lr ${LR} \
    --active_learning_method ${METHOD} \
    --num_active_learning_samples ${SIZE} \
    --num_active_learning_increments ${NUM_INC} \
    --pre_training 0 \
    --network_seed ${NETWORK_SEED} \
    --class_permutation_seed ${PERMUTATION_SEED} \
    --num_attribute_categories 253 \
    --num_predicate_categories 46 \
    --num_head_samples_per_class ${NUM_PRE_TRAIN_SAMPLES} \
    --ckpt_file ${CKPT} >${BASE_DIR}logs/${EXPT_NAME}.log
done
