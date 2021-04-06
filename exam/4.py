'''
from __future__ import print_function
import tensorflow as tf
import time
import numpy as np


# 세션을 실행한다.
sess = tf.InteractiveSession()

# 사이즈 100 큐를 생성하고 eqneue 노드를 정의한다.
# 임의의 값을 enqueue하는 enqueue_op 노드를 정의한다.
gen_random_normal = tf.random_normal(shape=())
queue = tf.train.shuffle_batch([gen_random_normal], batch_size=10, capacity=1000, min_after_dequeue=10)

# 10개의 쓰레드를 만들고 각각의 쓰레드가 병렬로(parallel) enqueue_op operation을 비동기적으로(asynchronous) 실행한다.
# 쓰레드를 컨트롤 할 수 있는 tf.train.Coordinator를 선언하고 각각의 쓰레드들을 tf.train.Coordinator에 넣어준다.

coord = tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess, coord=coord)
q=queue*10
print(queue)
# 10개의 쓰레드가 병렬적으로 연산을 수행한다.
# 아웃풋 예시 :
# 25
# 77
# 100

print(sess.run(tf.shape(queue)))
print(sess.run(queue))
print(sess.run(q))

coord.request_stop()
coord.join(threads)

'''
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import categorical_accuracy as accuracy
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input, Activation, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy   # tensorflow.keras.objectives는 tf 1.12에는 없는듯
import numpy as np



''' 
x = tf.placeholder(tf.float32, shape=(None, 448,448,3))
labels=tf.placeholder(tf.float32, shape=(None, 14,14))

model = ResNet50(input_tensor=x, include_top=False, weights=None, pooling='max')
model.summary()

model.trainable = True
selected_model=Model(inputs=model.input, outputs=model.get_layer('activation_48').output)     # output=(None, 7, 7, 2048)
y = selected_model.output
y=Conv2D(filters=512, strides=(1,1), kernel_size=(3,3), activation='relu', padding='same', name="C1")(y)
y=Conv2D(filters=128, strides=(1,1), kernel_size=(3,3), activation='relu', padding='same', name="C2")(y)
y=Conv2D(filters=32, strides=(1,1), kernel_size=(3,3), activation='relu', padding='same', name="C3")(y)
y=Conv2D(filters=8, strides=(1,1), kernel_size=(3,3), activation='relu', padding='same', name="C4")(y)
y=Conv2D(filters=1, strides=(1,1), kernel_size=(3,3), activation='relu', padding='same', name="C4")(y)
pred= tf.squeeze(y,squeeze_dims=3)

#tf.cast(labels, tf.float32)


img=np.random.rand(4,448,448,3)
img=np.array(img, dtype='float32')
label=np.array([[1,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 [1,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 [1,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 [1,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 ], dtype='float32')


im = np.empty((4, 14, 14), dtype=np.float32)
im[:, :, :] = label[np.newaxis,:, :]
label = im

loss = tf.reduce_mean(categorical_crossentropy(labels, pred))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

sess=tf.Session()
k.set_session(sess)

init_op = tf.global_variables_initializer()
sess.run(init_op)

for i in range(10000):
    _, cost=sess.run([train_step,loss], feed_dict={x: img, labels: label})
    if i%100==0:
        print(i)
        print(cost)

sess.close()
'''
a=np.array([[1,2],[3,4]])
b=a*10
print(b)