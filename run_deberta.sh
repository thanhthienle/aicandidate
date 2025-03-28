CUDA_VISIBLE_DEVICES=7 python run_deberta.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --cross_validate \
  --num_slices 4 \
  --train_file Data/hf_data.json \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 50 \
  --output_dir "output_deberta/" \
  --overwrite_output_dir \
  --logging_steps 1 \
  --logging_dir "output_deberta/" \
  --save_steps 100 \
  --overwrite_cache
