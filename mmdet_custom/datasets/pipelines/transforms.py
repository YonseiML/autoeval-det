import copy
import inspect

import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from imgaug.augmentables.polys import Polygon
from imgaug.augmentables.kps import KeypointsOnImage
import numpy as np
from imgaug import augmenters as iaa

import json
import time
import pycocotools.mask as maskUtils
import cv2
import random

import pycocotools.mask as cocomask
from  matplotlib import pyplot as plt
import os
import copy

import json


@PIPELINES.register_module()
class Metaaug:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        for key in results.get('img_fields', ['img']):
            low = 1
            high = 3
            aug_type = [
                iaa.imgcorruptlike.ElasticTransform(severity=random.randint(1,high)),
                iaa.imgcorruptlike.ImpulseNoise(severity=random.randint(1,high)),
                iaa.imgcorruptlike.ShotNoise(severity=random.randint(1,high)),
                iaa.imgcorruptlike.Contrast(severity=random.randint(1,high)),
                iaa.imgcorruptlike.GaussianNoise(severity=random.randint(1,high)),
                iaa.imgcorruptlike.SpeckleNoise(severity=random.randint(1,high)),
                iaa.imgcorruptlike.GaussianBlur(severity=random.randint(1,high)),
                iaa.imgcorruptlike.GlassBlur(severity=random.randint(1,high)),
                iaa.imgcorruptlike.DefocusBlur(severity=random.randint(1,high)),
                iaa.imgcorruptlike.MotionBlur(severity=random.randint(1,high)),
                iaa.imgcorruptlike.ZoomBlur(severity=random.randint(1,high)),
                iaa.imgcorruptlike.Fog(severity=random.randint(1,high)),
                iaa.imgcorruptlike.Frost(severity=random.randint(1,high)),
                iaa.imgcorruptlike.Snow(severity=random.randint(1,high)),
                iaa.imgcorruptlike.Spatter(severity=random.randint(1,high)),
                iaa.imgcorruptlike.Brightness(severity=random.randint(1,high)),
                iaa.imgcorruptlike.Saturate(severity=random.randint(1,high)),
                iaa.imgcorruptlike.JpegCompression(severity=random.randint(1,high)),
                iaa.imgcorruptlike.Pixelate(severity=random.randint(1,high)),
            ]

            aug_num = random.randint(1, 3) #len(aug_type))
            iaa_select_three = random.sample(aug_type, aug_num)
            seq = iaa.Sequential(iaa_select_three, random_order=True)  # apply augmenters in random order
            image_aug = seq.augment_image(results[key])
            results[key] = image_aug

            # results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
            #                                 self.to_rgb)
        # results['img_norm_cfg'] = dict(
        #     mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__

        return repr_str
