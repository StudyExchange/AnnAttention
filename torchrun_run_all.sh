#!/bin/bash


time_str=$(date +"%Y%m%d-%H%M00")


log_file="log/${time_str}.log"
mkdir -p log
touch "$log_file"
echo $log_file



epochs=1
max_seq_len=2024
batch_size=16
debug_step=20
gpu_nums=2

log_interval=100
save_interval=500
eval_interval=1000
if [[ "$debug_step" -ne 0 ]]; then
    log_interval=10
    save_interval=10
    eval_interval=20
fi



torchrun --nproc_per_node $gpu_nums train_pretrain.py  \
    --epochs  $epochs  \
    --max_seq_len $max_seq_len   \
    --batch_size $batch_size   \
    --debug_step $debug_step   \
    --log_interval $log_interval   \
    --save_interval $save_interval   \
    --eval_interval $eval_interval   \
    2>&1 | tee -a $log_file



torchrun --nproc_per_node $gpu_nums train_full_sft.py  \
    --epochs  $epochs  \
    --max_seq_len $max_seq_len   \
    --batch_size $batch_size   \
    --debug_step $debug_step   \
    --log_interval $log_interval   \
    --save_interval $save_interval   \
    --eval_interval $eval_interval   \
    2>&1 | tee -a $log_file



# python eval_model.py  --model_mode=0  \
#     2>&1 | tee -a $log_file
