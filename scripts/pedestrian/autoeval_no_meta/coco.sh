DATASET=coco
ANN_DIR=./data_pedestrian/${DATASET}/coco_annotations_sample_img_250.json

POS_ARRAY="1 2"
POS_NAME="1_2"
DROP=0.15
DETECTOR=$1
BACKBONE=$2
TASK=person

TAG1=cost_droprate_0_15_${DATASET}_droppos_${POS_NAME}_s250_n50
for i in `seq 0 0`;
  do
  if [ -f "./res_iou_ac_area_filter/${TAG1}/${i}.json" ]; then
    echo "pass"
  else
    echo "./res_iou_ac_area_filter/${TAG1}/${i}.json not exist"
    python tools/test_test.py configs/${DETECTOR}/${DETECTOR}_${BACKBONE}_fpn_1x_coco_${TASK}.py ./checkpoints/${DETECTOR}_${BACKBONE}_${TASK}/epoch_36.pth --eval bbox --cfg-option data.test.img_prefix=./data_pedestrian/${DATASET} data.test.ann_file=${ANN_DIR}  --tag1 ${TAG1} --tag2 ${i} --task ${TASK} --detector ${DETECTOR} --backbone ${BACKBONE}
  fi
done
ls -l ./res_iou_ac_area_filter/${TAG1}/* | grep "^-" | wc -l
