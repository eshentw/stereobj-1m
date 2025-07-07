

input_dir=log_object_triangulation_val
output_dir=log_object_triangulation_val
split=val


python baseline_keypose/merge_json_all_classes.py \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --split $split
