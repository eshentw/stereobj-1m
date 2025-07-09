
gpu=0
object_data=data/stereobj_1m/images_annotations/objects
root_dir=data/stereobj_1m/images_annotations/

##### for evaluating merged validation set performance
gt_json=data/stereobj_1m/images_annotations/val_label_merged.json # downloaded merged ground truth file
input_json=log_object_triangulation_val/merged.json # user input file
# input_json=../baseline_keypose/log_classic_triangulation_val/merged.json
# input_json=../baseline_keypose/log_object_triangulation_val/merged.json

##### INTERNAL USE ONLY: for evaluating merged test set performance
# gt_json=/path/to/test_label_merged.json # internal merged test set ground truth file
# input_json=../baseline_keypose/log_pnp_test/merged.json
# input_json=../baseline_keypose/log_classic_triangulation_test/merged.json
# input_json=../baseline_keypose/log_object_triangulation_test/merged.json


python evaluation/evaluate_pose6d.py \
    --gpu $gpu \
    --object_data $object_data \
    --input_json $input_json \
    --gt_json $gt_json \
    --root_dir $root_dir \
    --store_image
