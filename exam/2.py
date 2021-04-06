from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
# libs/datasets/dataset_factory.py
# from libs.visualization.summary_utils import visualize_input
import glob
import time
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from tensorflow.python import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import categorical_accuracy as accuracy
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input, Activation, Conv2D, Conv2DTranspose, Flatten, Dropout, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy

gt_boxes=np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]])
gt_box=gt_boxes[:,:4]
print(gt_box)

gt_box=gt_box/10
print(gt_box)
''''
QUEUE_LENGTH = 10

q = tf.FIFOQueue(QUEUE_LENGTH,"float")
enq_ops = q.enqueue_many(([1.0,2.0,3.0,4.0],) )
qr = tf.train.QueueRunner(q,[enq_ops,enq_ops,enq_ops])

sess = tf.Session()

# Create a coordinator, launch the queue runner threads.
coord = tf.train.Coordinator()
threads = qr.create_threads(sess, coord=coord, start=True)



for step in range(20):
    print(sess.run(q.dequeue()))



coord.request_stop()
coord.join(threads)
sess.close()



q=tf.train.shuffle_batch([], batch_size=10, capacity=100, min_after_dequeue=10)
'''

