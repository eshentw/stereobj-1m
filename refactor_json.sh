output_dir=output
gt_json=data/stereobj_1m/images_annotations/val_label_merged.json
split=val

python refactor_json.py \
    --gt_json $gt_json \
    --output_dir $output_dir \
    --split $split
