from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os


from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

'''
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

annFile = 'coco_instances_more-imgs.json'
coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['neck']);
imgIds = coco.getImgIds(catIds=catIds)

img = coco.loadImgs(imgIds[0])[0]

print(img)

I = io.imread(os.path.join('dataset/photo/photoresize', img['file_name']))
plt.axis('off')
plt.imshow(I)


plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()
'''