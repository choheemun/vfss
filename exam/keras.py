import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import categorical_accuracy as accuracy
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input, Activation, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy   # tensorflow.keras.objectives는 tf 1.12에는 없는듯
import numpy as np


x = tf.placeholder(tf.float32, shape=(None, 448,448,3))
labels=tf.placeholder(tf.float32, shape=(None, 56,56,1))

model = ResNet50(input_tensor=x, include_top=False, weights=None, pooling='max')
model.trainable = True
selected_model1=Model(inputs=model.input, outputs=model.get_layer('activation_48').output)     # output=(None, 14, 14, 2048)
selected_model2=Model(inputs=model.input, outputs=model.get_layer('activation_39').output)     # output=(None, 28, 28, 1024)
selected_model3=Model(inputs=model.input, outputs=model.get_layer('activation_21').output)     # output=(None, 56, 56, 512)
selected_model4=Model(inputs=model.input, outputs=model.get_layer('activation_9').output)     # output=(None, 112, 112, 256)
# selected_model1 ~ 3 으로만 pyramid를 build 하자

y1 = selected_model1.output
y2 = selected_model2.output
y3 = selected_model3.output
y4 = selected_model4.output

pyramid_1 = Conv2D(filters=256, strides=(1, 1), kernel_size=(1, 1), activation='relu', padding='same', name="P1")(y1)        #(None, 14, 14, 256)

pre_P2_1=tf.image.resize_bilinear(pyramid_1, [28,28], name='pre_P2_1')                                                     #(None, 28, 28, 256)
pre_P2_2 = Conv2D(filters=256, strides=(1, 1), kernel_size=(1, 1), activation='relu', padding='same', name="pre_p2_2")(y2)   #(None, 28, 28, 256)
pre_P2=tf.add(pre_P2_1,pre_P2_2)                                                                                             #(None, 28, 28, 256)
pyramid_2=Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), activation='relu', padding='same', name="P2")(pre_P2)      #(None, 28, 28, 256)

pre_P3_1=tf.image.resize_bilinear(pyramid_2, [56,56], name='pre_P3_1')                                                     #(None, 56, 56, 256)
pre_P3_2 = Conv2D(filters=256, strides=(1, 1), kernel_size=(1, 1), activation='relu', padding='same', name="pre_p3_2")(y3)   #(None, 56, 56, 256)
pre_P3=tf.add(pre_P3_1,pre_P3_2)                                                                                             #(None, 56, 56, 256)
pyramid_3=Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), activation='relu', padding='same', name="P3")(pre_P3)      #(None, 56, 56, 256)
m=pyramid_3

#pre_P4_1= =tf.image.resize_bilinear(pyramid_2, [112,112], name='pre_P4_1')                                                   #(None, 112, 112, 256)
#pre_P4_2 = Conv2D(filters=256, strides=(1, 1), kernel_size=(1, 1), activation='relu', padding='same', name="pre_p4_2")(y4)   #(None, 112, 112, 256)
#pre_P4=tf.add(pre_P4_1,pre_P4_2)                                                                                             #(None, 112, 112, 256)
#pyramid_4=Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), activation='relu', padding='same', name="P4")(pre_P4)      #(None, 112, 112, 256)

for _ in range(4):
   m = Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), activation='relu', padding='same', name="C1")(m)              #(None, 56, 56, 256)

m = Conv2D(filters=64, strides=(1, 1), kernel_size=(1, 1), activation='relu', padding='same')(m)                             #(None, 56, 56, 64)
m = Conv2D(filters=16, strides=(1, 1), kernel_size=(1, 1), activation='relu', padding='same')(m)                             #(None, 56, 56, 16)
mask = Conv2D(filters=1, strides=(1, 1), kernel_size=(1, 1), activation='sigmoid', padding='same', name="mask")(m)           #(None, 56, 56, 1)



'''
img=np.random.rand(1,448,448,3)
img=np.array(img, dtype='float32')
label=np.array([[[1,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 [1,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 [1,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 [1,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 ]], dtype='float32')

'''



loss = tf.reduce_mean(categorical_crossentropy(gt_mask, mask))       #(None, 56, 56, 1)  vs  (None, 56, 56, 1)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


sess=tf.Session()
k.set_session(sess)

init_op = tf.global_variables_initializer()
sess.run(init_op)

for i in range(10000):
    _, cost=sess.run([train_step,loss], feed_dict={x: img, labels: gt_mask})
    if i%100==0:
        print(i)
        print(cost)




sess.close()



