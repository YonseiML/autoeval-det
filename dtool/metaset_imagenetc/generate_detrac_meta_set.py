import os
import random

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import trange

import json
import cv2
import sys
sys.path.append(os.getcwd())
from imagenet_c import corrupt


try:
    os.makedirs('./data_inc/detrac/meta')
except:
    print('Alread has this path')

ROOT_DIR = "./data/detrac"
IMAGE_DIR = "./data/detrac/detrac2yolo/all"
TARGET_DIR = "./data_inc/detrac/meta"
num_meta_dataset = 50

with open(ROOT_DIR + "/detrac2coco/detrac2coco_sample_img_250.json",'r') as load_f:
    load_dict = json.load(load_f)

new_dict = {}
new_dict['images'] = list()
new_dict['annotations'] = list()
new_dict['categories'] = [{'supercategory': 'vehicle', 'id': 1, 'name': 'car'}]
img_list = []


new_dict['annotations'] = load_dict['annotations']
new_dict['images'] = load_dict['images']
       
print("car annotation", len(new_dict['annotations']))
print("car img", len(new_dict['images']))

car_image_id_dict = {}
for i in range(len(new_dict['images'])):
    if new_dict['images'][i]['file_name'] not in car_image_id_dict.keys():
        car_image_id_dict[new_dict['images'][i]['file_name']] = new_dict['images'][i]['id']
 
img_info = {}

for i in range(len(load_dict['images'])):
    img_info[load_dict['images'][i]['id']] = load_dict['images'][i]

tesize = len(car_image_id_dict.keys())


corruption_list = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'snow',
    'frost',
    'fog',
    'contrast',
    'pixelate',
    'jpeg_compression',
]

severities = [1.0, 2.0, 3.0, 4.0, 5.0]

for num in trange(num_meta_dataset):
    for img_path in list(car_image_id_dict.keys()):
        img_name = os.path.join(IMAGE_DIR, img_path)
        img = cv2.imread(img_name)
        original_size = (img.shape[1], img.shape[0]) 
        selected_corruption = corruption_list[num%10]
        severity = severities[num//10]
        corrupted_img = corrupt(img, corruption_name=selected_corruption, severity=severity)
        target_subdir = os.path.join(TARGET_DIR, str(num))
        os.makedirs(target_subdir, exist_ok=True)
        cv2.imwrite(os.path.join(target_subdir, img_path), corrupted_img)
        