from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import zipfile
import time
import numpy as np
import tensorflow as tf
from six.moves import urllib
from PIL import Image
import skimage.io as io
from matplotlib import pyplot as plt
from pycocotools.coco import COCO


from tensorflow.python.lib.io.tf_record import TFRecordCompressionType



class ImageReader(object):
  def __init__(self):
    self._decode_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_data, channels=3)
    self._decode_png = tf.image.decode_png(self._decode_data)

  def read_jpeg_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape

  def read_png_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
    return image


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
    output_filename = 'coco_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)


def _get_image_filenames(image_dir):
    return sorted(os.listdir(image_dir))


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _to_tfexample(image_data, image_format, label_data, label_format, height, width):
    """Encode only masks """
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_data),
        'image/format': _bytes_feature(image_format),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'label/encoded': _bytes_feature(label_data),
        'label/format': _bytes_feature(label_format),
        'label/height': _int64_feature(height),
        'label/width': _int64_feature(width),
    }))


def _to_tfexample_coco(image_data, image_format, label_data, label_format,
                       height, width,
                       num_instances, gt_boxes, masks):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_data),
        'image/format': _bytes_feature(image_format),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),

        'label/num_instances': _int64_feature(num_instances),  # N
        'label/gt_boxes': _bytes_feature(gt_boxes),  # of shape (N, 5), (x1, y1, x2, y2, classid)
        'label/gt_masks': _bytes_feature(masks),  # of shape (N, height, width)

        'label/encoded': _bytes_feature(label_data),  # deprecated, this is used for pixel-level segmentation
        'label/format': _bytes_feature(label_format),
    }))


def _to_tfexample_coco_raw(image_id, image_data, label_data,
                           height, width,
                           num_instances, gt_boxes, masks):
    """ just write a raw input"""
    return tf.train.Example(features=tf.train.Features(feature={
        'image/img_id': _int64_feature(image_id),
        'image/encoded': _bytes_feature(image_data),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'label/num_instances': _int64_feature(num_instances),  # N
        'label/gt_boxes': _bytes_feature(gt_boxes),  # of shape (N, 5), (x1, y1, x2, y2, classid)
        'label/gt_masks': _bytes_feature(masks),  # of shape (N, height, width)
        'label/encoded': _bytes_feature(label_data),  # deprecated, this is used for pixel-level segmentation
    }))


def _get_coco_masks(coco, img_id, height, width):       # 함수를 보면 gimp에서 어떤color로 masking하였는지는 상관없다는 것을 알수 있다.
                                                        # 이 함수를 통해 모든 mask 는 zero-one mask가 된다.
                                                        # colored mask(gimp) -> polygon(coco)-> zero-one mask(convert_coco)
    """ get the masks for all the instances
    Note: some images are not annotated
    Return:
      masks, mxhxw numpy array
      classes, mx1
      bboxes, mx4
    """
    annIds = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    # assert  annIds is not None and annIds > 0, 'No annotaion for %s' % str(img_id)
    anns = coco.loadAnns(annIds)
    # assert len(anns) > 0, 'No annotaion for %s' % str(img_id)
    masks = []
    classes = []
    bboxes = []
    mask = np.zeros((height, width), dtype=np.float32)
    segmentations = []
    for ann in anns:
        m = coco.annToMask(ann)  # zero one mask
        assert m.shape[0] == height and m.shape[1] == width, \
            'image %s and ann %s dont match' % (img_id, ann)
        masks.append(m)
        cat_id = ann['category_id']
        classes.append(cat_id)
        bboxes.append(ann['bbox'])
        m = m.astype(np.float32) * cat_id
        mask[m > 0] = m[m > 0]

    masks = np.asarray(masks)
    classes = np.asarray(classes)
    bboxes = np.asarray(bboxes)
    # to x1, y1, x2, y2
    #if bboxes.shape[0] <= 0:
    #    bboxes = np.zeros([0, 4], dtype=np.float32)
    #    classes = np.zeros([0], dtype=np.float32)
    #    print('None Annotations %s' % img_name)
    #    LOG('None Annotations %s' % img_name)
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    gt_boxes = np.hstack((bboxes, classes[:, np.newaxis]))
    gt_boxes = gt_boxes.astype(np.float32)
    masks = masks.astype(np.uint8)
    mask = mask.astype(np.uint8)
    assert masks.shape[0] == gt_boxes.shape[0], 'Shape Error'

    return gt_boxes, masks, mask


def _add_to_tfrecord(record_dir, image_dir, annotation_dir):
    """Loads image files and writes files to a TFRecord.
    Note: masks and bboxes will lose shape info after converting to string.--> 얘들은 decoding을 두번해야한다.
    """


    annFile = os.path.join(annotation_dir, 'coco_instances_more-imgs1.json')    # json파일명을 매번 수정하고
                                              # train.0~4py 각각 json 파일을 'annotation_dir' 즉 datasets/photo/annotation 으로 옮겨누어야함

    coco = COCO(annFile)

    cats = coco.loadCats(coco.getCatIds())
    #print('json file has %d images' % (len(coco.imgs)))
    imgs = [(img_id, coco.imgs[img_id]) for img_id in coco.imgs]    # imgs는 튜플인데 하나는 img_id 즉 scalar이고 나머지 하나는 cocodataset 'image' 항목의 모든 내용이다. license, file_name
                                                                    # height, width, id 가 여기에 속한다. 따라서 imgs[i][1][height]하면 i번째 img의 height 가 리턴
    #num_shards = int(len(imgs) / 2500)
    #num_per_shard = int(math.ceil(len(imgs) / float(num_shards)))   --> 본인 data를 수가 적으므로 2500개씩 나누지X

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        image_reader = ImageReader()

        # encode mask to png_string
        mask_placeholder = tf.placeholder(dtype=tf.uint8)       #  ????
        encoded_image = tf.image.encode_png(mask_placeholder)

        with tf.Session('') as sess:
            #for shard_id in range(num_shards):
            record_filename = os.path.join(record_dir,'coco_tfrecord')   # 결과로 나온 경로는 ./dataset/photo/coco_tfrecord/
            options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
            with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer: # tfrecord는 ./dataset/photo/ 에 coco_tfrecord 라는 이름으로 저장
                #start_ndx = shard_id * num_per_shard
                #end_ndx = min((shard_id + 1) * num_per_shard, len(imgs))
                for i in range(len(imgs)):
                    #if i % 50 == 0:
                    #    sys.stdout.write('\r>> Converting image %d/%d shard %d\n' % (
                    #        i + 1, len(imgs), shard_id))
                    #    sys.stdout.flush()

                    # image id and path
                    img_id = imgs[i][0]   # ex) 1
                    img_name = imgs[i][1]['file_name']     # ex) 1.png
                    #split = img_name.split('_')[1]   # 이름 중간_이 있는 경우 parsing 수정코드에선 의미x
                    img_name = os.path.join(image_dir, img_name) # img의 경로 './dataset/photo
                    ''' 
                    if FLAGS.vis:
                        im = Image.open(img_name)    ---> mask visualization 안하고 넘어가겠음
                        im.save('img.png')
                        plt.figure(0)
                        plt.axis('off')
                        plt.imshow(im)
                        # plt.show()
                        # plt.close()
                                      
                    # jump over the damaged images
                    #if str(img_id) == '320612':
                    #    continue
                    '''
                    # process anns
                    height, width = imgs[i][1]['height'], imgs[i][1]['width']    # cocodataset 에서 img의 정보중 h,w 가져오기
                    gt_boxes, masks, mask = _get_coco_masks(coco, img_id, height, width)

                    # read image as RGB numpy
                    img = np.array(Image.open(img_name))     # img 는 첫등장
                    if img.size == height * width:
                        print('Gray Image %s' % str(img_id))
                        im = np.empty((height, width, 3), dtype=np.uint8)
                        im[:, :, :] = img[:, :, np.newaxis]
                        img = im

                    img = img.astype(np.uint8)
                    assert img.size == width * height * 3, '%s' % str(img_id)

                    img_raw = img.tostring()
                    mask_raw = mask.tostring()

                    example = _to_tfexample_coco_raw(
                        img_id,
                        img_raw,   # img 그자체
                        mask_raw,   # mask 자체
                        height, width, gt_boxes.shape[0],
                        gt_boxes.tostring(), masks.tostring())

                    tfrecord_writer.write(example.SerializeToString())
sys.stdout.write('\n')
sys.stdout.flush()


def run():
    '''
    Runs the download and conversion operation.
    Args:
    dataset_dir: The dataset directory where the dataset is stored.
    '''
    dataset_dir = './dataset/photo'

    # for url in _DATA_URLS:
    #   download_and_uncompress_zip(url, dataset_dir)

    record_dir = os.path.join(dataset_dir, 'records')
    annotation_dir = os.path.join(dataset_dir, 'annotations')

    dataset_dir = './dataset/photo/photoresize'

    if not tf.gfile.Exists(record_dir):
        tf.gfile.MakeDirs(record_dir)

    # if not tf.gfile.Exists(annotation_dir):
    #    tf.gfile.MakeDirs(annotation_dir)     ---> 이건 할필요 없음 _add_to_tfrecord 내에서 dir + json 까지 설정해서 만듬

    # process the training, validation data:
    # if dataset_split_name in ['train2014', 'val2014']: 조건 필요없음 dataset중 val data 없음
    _add_to_tfrecord(record_dir,
                     dataset_dir,
                     annotation_dir)
    ''' val data 따로 안씀
        if dataset_split_name in ['trainval2014', 'minival2014']:
            _add_to_tfrecord_trainvalsplit(
                record_dir,
                dataset_dir,
                annotation_dir,
                dataset_split_name)

        print('\nFinished converting the coco dataset!')

    '''




if __name__ == '__main__':
        run()