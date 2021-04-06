import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

# mnist dataset 저장할 디렉터리
save_dir = 'exam/mnist'

# save_dir 에 MNIST 데이터 받기
data_sets = mnist.read_data_sets(save_dir,
                                 dtype=tf.uint8,
                                 reshape=False,
                                 validation_size=1000)

data_splits = ['train', 'test', 'validation']
for i, split in enumerate(data_splits):
    print("saving %s" % split)
    data_set = data_sets[i]

    filename = os.path.join(save_dir, '%s.tfrecords' % split)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(data_set.images.shape[0]):
        image = data_set.images[index].tostring()
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'height': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[data_set.images.shape[1]])),
                'width': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[data_set.images.shape[2]])),
                'depth': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[data_set.images.shape[3]])),
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[int(data_set.labels[index])])),
                'image_raw': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[image]))
            }))
        writer.write(example.SerializeToString())
    writer.close()

print('train, test, validation TFRecords saved!')

# READ Train dataset

NUM_EPOCHS = 10

filename = os.path.join(save_dir, 'train.tfrecords')
filename_queue = tf.train.string_input_producer([filename],
                                                num_epochs=NUM_EPOCHS)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

image = tf.decode_raw(features['image_raw'], tf.uint8)
image.set_shape([784])
image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
label = tf.cast(features['label'], tf.int32)
# 랜덤한 데이터(인스턴스)를 배치에 모은다.
images_batch, labels_batch = tf.train.shuffle_batch([image, label],
                                                    batch_size=128,
                                                    capacity=2000,
                                                    min_after_dequeue=1000)

W = tf.get_variable('W', [28 * 28, 10])
y_pred = tf.matmul(images_batch, W)

# loss
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                   labels=labels_batch))
# optimizer
global_step = tf.Variable(0, trainable=False, name='global_step')
train_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

with tf.Session() as sess:
    #init = tf.global_variables_initializer()
    #sess.run(init)
    #init = tf.local_variables_initializer()
    #sess.run(init)
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=20)
    ckpt = tf.train.get_checkpoint_state('exam/mnist')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)         # 1.단순히 epoch를 연장하여 학습을 이어가는 경우
        #saver.restore(sess, "./mnist/nn.ckpt-3500")              # 2.overfitting 이 확인되어 특정 epoch를 선택하여 학습을 이어가는 경우
                                                                 # 1 or 2 중 하나만 실행시키면 된다.
        sess.run(tf.local_variables_initializer())       # string_input_producer 의 epoch 때문...

    else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    # coordinator
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(threads)
    try:
        step=0
        while not coord.should_stop():
            step += 1
            sess.run([train_op])  # feed_dict를 쓰지 않는다.
            if step % 500 == 0:
                loss_val = sess.run(loss)
                print('Step: {:4d} | Loss: {:.5f}'.format(step, loss_val))
                checkpoint_path=os.path.join('exam/mnist', 'nn.ckpt')
                saver.save(sess,checkpoint_path, global_step = global_step)

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
    finally:
        # 완료되면 스레드 중지를 요청한다.
        coord.request_stop()

    # 스레드가 완료되길 기다린다.
    coord.join(threads)

    # example -- get image,label
    # img1, lbl1 = sess.run([image, label])

    # example - get random batch
    # labels, images = sess.run([labels_batch, images_batch])