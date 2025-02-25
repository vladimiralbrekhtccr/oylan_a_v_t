#!/bin/bash
set -x

eval "$(conda shell.bash hook)"
conda activate oylan_a_v_t
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Basic configuration
GPUS=${GPUS:-8}
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACC=1
NNODES=6
MASTER_IP="10.141.0.1"  # Node 8 IP
MASTER_PORT=32225

# Important: Change to the correct working directory first
cd /scratch/vladimir_albrekht/projects/oylan_a_v_t/internvl_omni_training/internvlomni_chat

export WANDB_PROJECT=internomni_us_cluster_6nodes_79B
export WANDB_API_KEY="9c677fae879db422dc5d598f4bcdac4c4e15f667"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=32225
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='/scratch/vladimir_albrekht/projects/oylan_a_v_t/output/train_1_oylan_a_v_t_79B_with_replaced_whisper_turbo_mlp2'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 32
# batch size per gpu: 1
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=${NNODES} \
  --node_rank=5 \
  --master_addr=${MASTER_IP} \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/scratch/vladimir_albrekht/projects/oylan_a_v_t/models/train_1_oylan_a_v_t_79B_with_replaced_whisper_turbo_mlp2_checkpoint-10000" \
  --conv_style "internomni" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/scratch/vladimir_albrekht/projects/oylan_a_v_t/data/meta_files/train_1_test.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.4 \
  --freeze_llm True \
  --freeze_backbone True \
  --freeze_mlp_vision True \
  --freeze_audio_encoder False \
  --freeze_mlp_audio False \
  --vision_select_layer -1 \
  --dataloader_num_workers 16 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 5000 \
  --save_total_limit 50 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail False \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config_100b.json" \
  --report_to "wandb" \
  --run_name "train_1_oylan_a_v_t_79B_with_replaced_whisper_turbo_mlp2_test1" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
