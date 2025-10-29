DETECTOR=$1
BACKBONE=$2

bash scripts/vehicle/autoeval_no_meta/bdd.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_no_meta/cityscapes.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_no_meta/coco.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_no_meta/detrac.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_no_meta/exdark.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_no_meta/kitti.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_no_meta/kaggle/self_driving.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_no_meta/kaggle/roboflow.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_no_meta/kaggle/traffic.sh "$DETECTOR" "$BACKBONE"
bash scripts/vehicle/autoeval_no_meta/kaggle/udacity.sh "$DETECTOR" "$BACKBONE"
