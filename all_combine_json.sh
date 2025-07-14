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

output_dir=log_object_triangulation_val
split=val

for i in "${!classes[@]}"; do
    cls_type="${classes[i]}"
    input_dir=log_object_triangulation_val/${cls_type}
    echo "Combining JSON for class: $cls_type"
    

    python baseline_keypose/object_triangulation_combine_json.py \
        --input_dir $input_dir \
        --output_dir $output_dir \
        --split $split \
        --cls_type $cls_type

done

