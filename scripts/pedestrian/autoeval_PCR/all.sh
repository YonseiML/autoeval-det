DETECTOR=$1
BACKBONE=$2

bash scripts/pedestrian/autoeval_PCR/citypersons.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_PCR/cityscapes.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_PCR/coco.sh "$DETECTOR" "$BACKBONE"  
bash scripts/pedestrian/autoeval_PCR/crowdhuman.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_PCR/ExDark.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_PCR/kitti.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_PCR/self_driving.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_PCR/caltech.sh "$DETECTOR" "$BACKBONE"
bash scripts/pedestrian/autoeval_PCR/ECP.sh "$DETECTOR" "$BACKBONE"
