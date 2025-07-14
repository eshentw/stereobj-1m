gpu=0
root_dir=data/stereobj_1m/images_annotations/
out_dir=visualizations
##### for evaluating merged validation set performance
refactor_gt_json=/home/eddie880509/src/stereobj-1m/output/refactor_gt_pred.json # downloaded merged ground truth file
input_json=log_object_triangulation_val/merged.json # user input file


python evaluation/draw_and_evalpose6d.py \
    --gpu $gpu \
    --input_json $input_json \
    --refactor_gt_json $refactor_gt_json \
    --root_dir $root_dir \
    --out_dir $out_dir \
    --store_image