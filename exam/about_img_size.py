import numpy as np


'''
filename_queue = tf.train.string_input_producer(["1", "2", "3"], shuffle=True)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for step in range(10):
        print(sess.run(filename_queue.dequeue()))

    coord.request_stop()
    coord.join(threads)

'''

from shapely.geometry import Polygon, MultiPolygon


import numpy as np
from PIL import Image
import os



img=os.path.join('./p','vfss2.png')
im = Image.open(img)

pix = np.array(im)

print(im.size)

print(pix[100][100])
print('-'*50)

img=os.path.join('./p','vfss2-1.png')   # shape가 (h,w,3)이었던 vfss1의 mask 인 vfss1-0은 shape가 (h,w,4)  3+1
                                        # shape가 (h,w)이었던 vfss2의 mask 인 vfss2-0은 shape가 (h,w,2)     1+1
im = Image.open(img)

pix = np.array(im)

print(im.size)

print(pix[100][100])
print('-'*50)

img=os.path.join('./p','vfss2-10.png')
im = Image.open(img)
pix = np.array(im)
print('2-12 auto')     # gimp 에서 '이미지 내보내기'를 auto 로 하면 depth+=1 이 된다.
print(im.size)
print(pix[100][100])
print('-'*50)

img=os.path.join('./p','vfss2-11.png')
im = Image.open(img)
pix = np.array(im)
print('2-12 rgb')
print(im.size)
print(pix[100][100])
print('-'*50)

img=os.path.join('./p','vfss2-12.png')
im = Image.open(img)
pix = np.array(im)
print('2-12 gray')
print(im.size)
print(pix[100][100])
print('-'*50)

img=os.path.join('./p','vfss2-13.png')
im = Image.open(img)
pix = np.array(im)
print('2-13 auto')
print(im.size)
print(pix[100][100])
print('-'*50)

img=os.path.join('./p','vfss2-14.png')
im = Image.open(img)
pix = np.array(im)
print('2-14 rgb')
print(im.size)
print(pix[100][100])
print('-'*50)

img=os.path.join('./p','vfss2-15.png')
im = Image.open(img)
pix = np.array(im)
print('2-15 gray')
print(im.size)
print(pix[100][100])
print('-'*50)



print('-'*50)
img=os.path.join('./p','vfss1-1.png')
im = Image.open(img)
#im=np.array(im)
print(im.shape)
pix = np.array(im)

print(im.size)

print(pix[100][100])
print('-'*50)


img=os.path.join('./p','1.png')
im = Image.open(img)

pix = np.array(im)

print(im.size)

print(pix[100][100])
print(pix[100][150])
print(pix[200][100])
print('-'*50)

img=os.path.join('./p','1-1.png')
im = Image.open(img)

pix = np.array(im)

print(im.size)

print(pix[100][100])
print(pix[100][150])
print(pix[200][100])

print('-'*50)
a=np.array([[1,2],[3,4],[5,6]])
print(a)
b=a.shape[0]
c=a.shape[1]
print(a.shape)
print(b)
print(c)








'''
from PIL import Image

image1 = Image.open('./p/1-1.png')   # 증명사진
imag1_size = image1.size
print(imag1_size)

image1=np.array(image1)  # 증명사진을 array로 변환
print(image1.shape)
print(image1.size)   # array로 바꾸면 h*w*d 가 size가 된다 즉 size는 scalar다
print('-'*50)


image2 = Image.open('./p/vfss2.png')    # vfss 사진    주의 "vfss1.shape=(h,w,3) 이고, vfss2.shape=(h,w)이다. 
                                        # vfss1은 rgb이고 vfss2는 grayscale 이다. 
imag2_size = image2.size
(height, width)=image2.size
print(imag2_size)
print(height, width)
image2=np.array(image2)
print(image2.shape)
print(image2.size)
print('-'*50)


image2=np.array(image2)
print(image2.shape)
print('-'*50)
im = np.empty((width, height, 3), dtype=np.uint8)   # height, width 순서에 유의
print(im.shape)
im[:, :, :] = image2[:, :, np.newaxis]
img = im
#img.show()

img_size = img.size

print(img_size)
'''