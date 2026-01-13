#!/bin/bash

model_name = "Test_0"
unknown_threshold=1
max_bgc_length=1024
d_model=128
n_layers=2
heads=8
dropout=0.1
batch_size=64
training_datasets_file = ""
seed=42
thread_count=8

cd C:\Users\JCboo\Documents\GitHub\BGC-MLM
python BGC_MLM_train.py \
    model_name $model_name \
    training_datasets_file $training_datasets_file \
    unknown_threshold $unknown_threshold \
    max_bgc_length $max_bgc_length \
    d_model $d_model \
    n_layers $n_layers \
    heads $heads \
    dropout $dropout \
    batch_size $batch_size \
    --seed $seed \
    --thread_count $thread_count