DATASET=kitti
ANN_DIR=./data/${DATASET}/kitti2coco/kitti2coco_sample_img_250.json



TAG1=${DATASET}_s250_n50
for i in `seq 0 49`;
  do
  if [ -f "./res_iou_ac_area_filter/${TAG1}/${i}.json" ]; then
    echo "pass"
  else
    echo "./res_iou_ac_area_filter/${TAG1}/${i}.json not exist"
    python tools/test.py configs/retinanet/retinanet_r50_fpn_1x_coco_car.py ./checkpoints/r50_retina_car/epoch_36.pth --eval bbox --cfg-option data.test.img_prefix=./data_inc/${DATASET}/meta/${i} data.test.ann_file=${ANN_DIR}  --tag1 ${TAG1} --tag2 ${i} 
  fi
done
ls -l ./res_iou_ac_area_filter/${TAG1}/* | grep "^-" | wc -l
