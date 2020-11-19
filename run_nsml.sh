#!/usr/bin/env bash
nsml run \
  -m 'kaist korquad open' \
  -d korquad-open-ldbd3 \
  -g 1 \
  -c 1 \
  -e run_squad.py \
  -a "--model_type electra
    --model_name_or_path monologg/koelectra-base-v2-finetuned-korquad
    --do_train
    --do_eval
    --data_dir train
    --num_train_epochs 2
    --per_gpu_train_batch_size 24
    --per_gpu_eval_batch_size 24
    --output_dir output
    --verbose_logging
    --overwrite_output_dir
    --version_2_with_negative"
