from coco import COCO
import json
from matplotlib.collections import PathCollection
import numpy as np
import copy
import itertools
#from . import mask as maskUtils
import os
from collections import defaultdict
import sys

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def coco_labels (anno_path) :
    coco = COCO(anno_path) 
    annotations = coco.anns 
    
    for anno in annotations :
        anno_dict = annotations[anno]
        anno_img_id = str(anno_dict['image_id']).zfill(12)
        print(anno_img_id)
        f = open('/home/youngjin/datasets/coco/val/labels/'+anno_img_id+'.txt','a')
        f.write(str(anno_dict['category_id']))
        f.write(' ')
        f.write(str(anno_dict['bbox'][0]))
        f.write(' ')
        f.write(str(anno_dict['bbox'][1])) 
        f.write(' ')
        f.write(str(anno_dict['bbox'][2])) 
        f.write(' ')
        f.write(str(anno_dict['bbox'][3])) 
        f.write('\n')
        f.close()


if __name__ == "__main__" :

    coco_labels = coco_labels('/home/youngjin/datasets/coco/annotations/instances_val2017.json')

    