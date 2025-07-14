#!/bin/bash

classes=(
    "blade_razor"
    # "hammer" 
    "needle_nose_pliers"
    "screwdriver"
    "side_cutters"
    "tape_measure"
    "wire_stripper"
    "wrench"
    # "centrifuge_tube"
    "microplate"
    "tube_rack_1.5_2_ml"
    "tube_rack_50_ml"
    "pipette_0.5_10"
    "pipette_10_100"
    "pipette_100_1000"
    "sterile_tip_rack_10"
    "sterile_tip_rack_200"
    "sterile_tip_rack_1000"
)

model_weights=(
    "model-16.ckpt" # blade_razor
    # "model-14.ckpt" # hammer
    "model-15.ckpt" # needle_nose_pliers
    "model-19.ckpt" # screwdriver
    "model-13.ckpt" # side_cutters
    "model-16.ckpt" # tape_measure
    "model-14.ckpt" # wire_stripper
    "model-17.ckpt" # wrench
    # "???" # centrifuge_tube
    "model-17.ckpt" # microplate
    "model-15.ckpt" # tube_rack_1.5_2_ml
    "model-14.ckpt" # tube_rack_50_ml
    "model-20.ckpt" # pipette_0.5_10
    "model-26.ckpt" # pipette_10_100
    "model-16.ckpt" # pipette_100_1000
    "model-15.ckpt" # sterile_tip_rack_10
    "model-24.ckpt" # sterile_tip_rack_200
    "model-25.ckpt" # sterile_tip_rack_1000
)

gpu=0
split=val
batch_size=1
model=model_res34_backbone
num_kp=16
data=data/stereobj_1m/images_annotations
dataset=stereobj1m_dataset
image_width=640
image_height=640
debug=0

for i in "${!classes[@]}"; do
    cls_type="${classes[i]}"
    model_weight="${model_weights[i]}"
    model_path="data/stereobj_1m/keypose_weights/log_train_${cls_type}/${model_weight}"

    echo "Evaluating class: $cls_type with weight: $model_weight"
    
    python baseline_keypose/evaluate_lr.py \
        --gpu $gpu \
        --batch_size $batch_size \
        --split $split \
        --model $model \
        --model_path "$model_path" \
        --cls_type "$cls_type" \
        --data "$data" \
        --dataset "$dataset" \
        --num_kp "$num_kp" \
        --image_width "$image_width" \
        --image_height "$image_height" \
        --debug "$debug"
done

