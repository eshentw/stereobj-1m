num_kp=16
cls_type=hammer
image_width=640
image_height=640
log_dir=log
split=val


python baseline_keypose/object_triangulation.py \
    --cls_type $cls_type \
    --num_kp $num_kp \
    --image_width $image_width \
    --image_height $image_height \
    --split $split
    # > $log_dir.txt 2>&1 &
