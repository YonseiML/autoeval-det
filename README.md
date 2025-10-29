# Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability [ICCV 2025 Oral]

[![Conference](https://img.shields.io/badge/ICCV-2025%20Oral-0b5fff.svg)]()
[![Paper](https://img.shields.io/badge/Paper-arXiv-4b9e5d.svg)](https://arxiv.org/abs/2508.12082)

This repository provides an official implementation of our ICCV 2025 Oral paper:
> [**Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability**](https://arxiv.org/abs/2508.12082)  
> **Seungju Yoo**, **Hyuk Kwon**, **Joong-Won Hwang**, and **Kibok Lee**

```BibTeX
@inproceedings{yoo2025automated,
  title={Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability},
  author={Yoo, Seungju and Kwon, Hyuk and Hwang, Joong-Won and Lee, Kibok},
  booktitle={ICCV},
  year={2025}
}
```

## Installation
```python
conda env create --file environment.yaml
mim install mmcv-full==1.7.2
```

## Data Preparation
### Vehicle Detection
We use the same dataset split as [[BoS](https://github.com/YangYangGirl/BoS)].
You can download their split [[here](https://drive.google.com/file/d/1bs1y04q_0VeSDTnex0i94gzK8vGXdx5r/view?usp=sharing)].
After downloading, unzip and place the content as `{PROJECT_DIR}/data`.

### Pedestrian Detection
We standardize nine existing object detection datasets: **COCO**, **Caltech**, **Cityscapes**, **Citypersons**, **Crowdhuman**, **ECP**, **ExDark**, **KITTI**, and **Self-driving**, as done for vehicle detection.
For each domain, 250 images containing pedestrians are randomly selected. Dataset splits in `.json` format can be found under `/data_pedestrian/{DATASET_NAME}`.
Download each dataset and place all files under the corresponding folder: `/data_pedestrian/{DATASET_NAME}`.

## Model Preparation
Each experiment evaluates a single-class detector trained for either vehicle or pedestrian detection.
You can either:
* Train your detectors using configurations under `/configs/_base_/models` and place the last checkpoints `epoch_36.pth` in `{PROJECT_DIR}/checkpoints/`.
* Download our checkpoints used in the paper [[here](https://www.dropbox.com/scl/fo/d4oejv2cok1iegepwa3ty/AH0Xg-63VD-A4bqQHkKJLiU?rlkey=a7z8lifwocd2kabi6hrok9e5r&st=uaeilotw&dl=0)].

## Generating Meta-Dataset
```python
# Vehicle 
bash scripts/metaset_generate_inc/all.sh

# Pedestrian
bash scripts/pedestrian/metaset_generate_inc_person/all.sh
```

## Computing PCR Score for AutoEval

```python
# Vehicle detection (meta-set)
bash scripts/vehicle/autoeval_PCR/all.sh [DETECTOR] [BACKBONE]

# For Retinanet+R50 variant
bash scripts/vehicle/autoeval_PCR/all.sh retinanet r50

# Vehicle detection (test set)
bash scripts/vehicle/autoeval_no_meta/all.sh [DETECTOR] [BACKBONE]

# For Retinanet+R50 variant
bash scripts/vehicle/autoeval_no_meta/all.sh retinanet r50

# Pedestrian detection (meta-set)
bash scripts/pedestrian/autoeval_PCR/all.sh [DETECTOR] [BACKBONE]

# For Retinanet+R50 variant
bash scripts/pedestrian/autoeval_PCR/all.sh retinanet r50

# Pedestrian detection (test set)
bash scripts/pedestrian/autoeval_no_meta/all.sh [DETECTOR] [BACKBONE]

# For Retinanet+R50 variant
bash scripts/pedestrian/autoeval_no_meta/all.sh retinanet r50
```
To evaluate using a different model, modify the config accordingly.

* `[DETECTOR] : retinanet, faster_rcnn`
* `[BACKBONE] : r50, swin`.

## Computing RMSE
```python
bash run_rmse.sh
```

## Acknowledgement
This work builds upon the following open-source projects:
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [BoS](https://github.com/YangYangGirl/BoS)
