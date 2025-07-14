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


num_kp=16
image_width=640
image_height=640
split=val

for i in "${!classes[@]}"; do
    cls_type="${classes[i]}"

    echo "Triangulate class: $cls_type"
    
    python baseline_keypose/object_triangulation.py \
        --cls_type $cls_type \
        --num_kp $num_kp \
        --image_width $image_width \
        --image_height $image_height \
        --split $split
done

