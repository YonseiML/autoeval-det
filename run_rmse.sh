list1=("coco" "bdd" "cityscapes" "detrac" "exdark" "kitti" "self_driving2coco" "roboflow2coco" "udacity2coco" "traffic2coco") # car
# list1=("coco" "caltech" "citypersons" "cityscapes" "crowdhuman" "ECP" "ExDark" "kitti" "self_driving") # person

numbers=(0)

for data in "${list1[@]}"
do
    for j in "${numbers[@]}"
    do
        echo "Data: $data, Original index: $j"
        CUDA_VISIBLE_DEVICES=0 python tools/rmse.py --target $data --orig_index $j
    done
done
