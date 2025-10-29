DATASET=cityscapes
ANN_DIR=./data/${DATASET}/cityscapes2coco/instancesonly_filtered_gtFine_val_car_sample_img_250.json

DETECTOR=$1
BACKBONE=$2
TASK=car
TAG1=${DATASET}_s250_n50

for i in `seq 0 0`;
  do
  if [ -f "./res/${TAG1}/${i}.json" ]; then
    echo "pass"
  else
    echo "./res/${TAG1}/${i}.json not exist"
    python tools/test_test.py configs/${DETECTOR}/${DETECTOR}_${BACKBONE}_fpn_1x_coco_${TASK}.py ./checkpoints/${DETECTOR}_${BACKBONE}_${TASK}/epoch_36.pth --eval bbox --cfg-option data.test.img_prefix=./data/${DATASET}/leftImg8bit/val data.test.ann_file=${ANN_DIR}  --tag1 ${TAG1} --tag2 ${i} --task ${TASK} --detector ${DETECTOR} --backbone ${BACKBONE}
  fi
done
ls -l ./res/${TAG1}/* | grep "^-" | wc -l
