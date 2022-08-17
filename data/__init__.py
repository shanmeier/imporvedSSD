# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES, VOCDetection_mosaic
from .voc0712 import VOCDetection_data_aug, AnnotationTransform_origin_SSD, VOCDetection_cachedir
from .hrrsd2017 import HRRSDAnnotationTransform, HRRSDDetection
from .coco import COCODetection
from .data_augment import *
from .config import *
