DATASET=cityscapes
ANN_DIR=./data_pedestrian/${DATASET}/cityscapes_annotations_sample_img_250.json

POS_ARRAY="1 2"
POS_NAME="1_2"
DROP=0.15

TAG1=cost_droprate_0_15_${DATASET}_droppos_${POS_NAME}_s250_n50
for i in 0 49;
  do
  if [ -f "./res/${TAG1}/${i}.json" ]; then
    echo "pass"
  else
    echo "./res/${TAG1}/${i}.json not exist"
    python tools_person/test_bos.py configs/retinanet/retinanet_r50_fpn_1x_coco_person.py ./checkpoints/r50_retina_person/epoch_36.pth --eval bbox --cfg-option data.test.img_prefix=./data_inc_pedestrian/${DATASET}/meta/${i} data.test.ann_file=${ANN_DIR}  --tag1 ${TAG1} --tag2 ${i} --dropout_uncertainty ${DROP} --drop_layers ${POS_ARRAY}
  fi
done
ls -l ./res/${TAG1}/* | grep "^-" | wc -l
