DETECTOR=$1
BACKBONE=$2

bash scripts/vehicle/autoeval_PCR/bdd.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_PCR/cityscapes.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_PCR/coco.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_PCR/detrac.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_PCR/exdark.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_PCR/kitti.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_PCR/kaggle/self_driving.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_PCR/kaggle/roboflow.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_PCR/kaggle/traffic.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_PCR/kaggle/udacity.sh "$DETECTOR" "$BACKBONE"
