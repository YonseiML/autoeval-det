DETECTOR=$1
BACKBONE=$2

bash scripts/vehicle/autoeval_bos/bdd.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_bos/cityscapes.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_bos/coco.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_bos/detrac.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_bos/exdark.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_bos/kitti.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_bos/kaggle/self_driving.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_bos/kaggle/roboflow.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_bos/kaggle/traffic.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_bos/kaggle/udacity.sh "$DETECTOR" "$BACKBONE"
