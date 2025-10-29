DATASET=traffic2coco
ANN_DIR=./data/kaggle/traffic2coco/traffic2coco_sample_img_250.json

DETECTOR=$1
BACKBONE=$2
TASK=car
TAG1=${DATASET}_s250_n50


for i in `seq 0 0`;
  do
  if [ -f "./res_iou_ac_area_filter/${TAG1}/${i}.json" ]; then
    echo "pass"
  else
    echo "./res_iou_ac_area_filter/${TAG1}/${i}.json not exist"
    python tools/test_test.py configs/${DETECTOR}/${DETECTOR}_${BACKBONE}_fpn_1x_coco_${TASK}.py ./checkpoints/${DETECTOR}_${BACKBONE}_${TASK}/epoch_36.pth --eval bbox --cfg-option data.test.img_prefix=./data/kaggle/${DATASET}/traffic_kaggle/train/images data.test.ann_file=${ANN_DIR}  --tag1 ${TAG1} --tag2 ${i} --task ${TASK} --detector ${DETECTOR} --backbone ${BACKBONE}
done
ls -l ./res_iou_ac_area_filter/${TAG1}/* | grep "^-" | wc -l
