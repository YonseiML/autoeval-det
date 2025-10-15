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

# for replication
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
ia.seed(SEED)

aug_type = [
        iaa.pillike.EnhanceColor(),  # remove a random fraction of color from input images
        iaa.Sharpen(alpha=(0.0, 1.0)),  # apply a sharpening filter kernel to images
        iaa.ChangeColorTemperature((1100, 10000 // 2)),  # change the temperature to a provided Kelvin value.
        iaa.pillike.Equalize(),  # equalize the image histogram
        iaa.Solarize(0.5, threshold=(32, 128)),  # invert the colors
        iaa.Multiply((0.8, 1.2), per_channel=0.2),  # make some images brighter and some darker
        iaa.pillike.Autocontrast(),  # adjust contrast by cutting off p% of lowest/highest histogram values
        iaa.Grayscale(alpha=(0.0, 0.5)),  # remove colors with varying strength
    ]
try:
    os.makedirs('./data/bdd/meta')
except:
    print('Alread has this path')

ROOT_DIR = "./data/bdd"
IMAGE_DIR = "./data/bdd/bdd100k/images/100k/val"
TARGET_DIR = "./data/bdd/meta"
num_meta_dataset = 50

with open(ROOT_DIR + "/bdd100k/labels_coco/bdd100k_labels_images_val_coco_car_sample_img_250.json",'r') as load_f:
    load_dict = json.load(load_f)

new_dict = {}
new_dict['images'] = list()
new_dict['annotations'] = list()
new_dict['categories'] = [{'supercategory': 'vehicle', 'id': 1, 'name': 'car'}]
img_list = []

for i in range(len(load_dict['annotations'])):
    if load_dict['annotations'][i]['category_id'] == 1:
        new_dict['annotations'].append(load_dict['annotations'][i])

car_images_id = []
for ann in new_dict['annotations']:
    if ann['image_id'] not in car_images_id:
        car_images_id.append(ann['image_id'])

for img in load_dict['images']:
    if img['id'] in car_images_id:
        new_dict['images'].append(img)
        
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

for num in trange(num_meta_dataset):
    for img_path in list(car_image_id_dict.keys()):
        img_name = os.path.join(IMAGE_DIR, img_path)
        img = cv2.imread(img_name)
        num_sel = 3
        
        list_sel = random.sample(aug_type, int(num_sel))
        random.shuffle(list_sel)
        seq = iaa.Sequential(list_sel, random_order=False, random_state=SEED + num)
        new_data = seq.augment_image(img)
    
        if not os.path.exists(os.path.join(TARGET_DIR, str(num))):
            os.mkdir(os.path.join(TARGET_DIR, str(num)))

        cv2.imwrite(os.path.join(TARGET_DIR, str(num), img_path), new_data)
   