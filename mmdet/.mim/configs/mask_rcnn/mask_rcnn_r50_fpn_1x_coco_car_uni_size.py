_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_car.py',
    '../_base_/datasets/coco_detection_car_uni_size.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)