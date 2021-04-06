import threading
import time
import tensorflow as tf
from pprint import pprint
import time

from tensorflow import keras

'''
sess = tf.InteractiveSession()

# RandomShuffleQueue 넣을 데이터 생성: -0.81226367과 같이 하나의 값만 출력됨
gen_random_normal = tf.random_normal(shape=())
# RandomShuffleQueue 정의
queue = tf.RandomShuffleQueue(capacity=10, dtypes=[tf.float32],
                              min_after_dequeue=1)
enqueue_op = queue.enqueue(gen_random_normal)

# QueueRunner를 이용한 멀티스레드 구현
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
#start=True 이므로 thread생성과 동시에 작동한다.

print(sess.run(queue.size()))
time.sleep(0.0001)
print(sess.run(queue.size()))
time.sleep(0.0001)
print(sess.run(queue.size()))
time.sleep(0.0001)
print(sess.run(queue.size()))

print('-'*100)


#a=tf.stack([1, queue.dequeue()])    #
a=tf.stack([1, gen_random_normal]) # 둘은 같은 processing 이다.
for i in range(200):
    print(sess.run(a))
coord.request_stop()
coord.join(enqueue_threads)


print(sess.run(queue.size()))
print('-'*100)
print(sess.run(queue.dequeue()))
print(queue.dequeue())
print(queue.dequeue())
print(queue.dequeue())
print(queue.dequeue())
print(queue.dequeue())
print(sess.run(queue.size()))

m=0
for _ in range(4):
    m+=1
print(m)
'''


a=[(256, 64, 1)] *2 + [(256, 64, 2)]
for i in a:
    print(i)