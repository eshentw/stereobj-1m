

cls_type=hammer
input_dir=log_object_triangulation_val/${cls_type}
output_dir=log_object_triangulation_val
split=val


python baseline_keypose/object_triangulation_combine_json.py \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --split $split \
    --cls_type $cls_type
