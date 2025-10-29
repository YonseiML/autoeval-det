DETECTOR=$1
BACKBONE=$2

bash scripts/pedestrian/autoeval_bos/citypersons.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_bos/cityscapes.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_bos/coco.sh "$DETECTOR" "$BACKBONE"  
bash scripts/pedestrian/autoeval_bos/crowdhuman.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_bos/ExDark.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_bos/kitti.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_bos/self_driving.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_bos/caltech.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_bos/ECP.sh "$DETECTOR" "$BACKBONE"
