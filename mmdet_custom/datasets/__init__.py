

from .coco_car import CocoCarDataset
from .coco_person import CocoPersonDataset
from .coco_person_and_car import CocoPersonAndCarDataset
from .coco_common_class import CocoCommonClassDataset
from .pipelines import Metaaug
__all__ = [
    'CocoCarDataset', 'CocoPersonDataset', 'CocoPersonAndCarDataset', 'CocoCommonClassDataset', 'Metaaug'
]