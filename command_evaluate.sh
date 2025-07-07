#!/bin/bash
command_file=`basename "$0"`
gpu=0
cls_type=sterile_tip_rack_10
model_path=data/TOD/keypose_weights/log_train_sterile_tip_rack_10/model-15.ckpt
split=val
batch_size=4
model=model_res34_backbone
num_kp=16
data=data/stereobj_1m/images_annotations
dataset=stereobj1m_dataset
image_width=640
image_height=640


python baseline_keypose/evaluate.py \
    --gpu $gpu \
    --batch_size $batch_size \
    --split $split \
    --model $model \
    --model_path $model_path \
    --cls_type $cls_type \
    --data $data \
    --dataset $dataset \
    --num_kp $num_kp \
    --image_width $image_width \
    --image_height $image_height \
    --command_file $command_file \
    # > $log_dir.txt 2>&1 &
