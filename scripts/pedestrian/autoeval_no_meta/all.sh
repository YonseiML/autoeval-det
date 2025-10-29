DETECTOR=$1
BACKBONE=$2

bash scripts/pedestrian/autoeval_no_meta/citypersons.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_no_meta/cityscapes.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_no_meta/coco.sh "$DETECTOR" "$BACKBONE"  
bash scripts/pedestrian/autoeval_no_meta/crowdhuman.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_no_meta/ExDark.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_no_meta/kitti.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_no_meta/self_driving.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_no_meta/caltech.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_no_meta/ECP.sh "$DETECTOR" "$BACKBONE"
