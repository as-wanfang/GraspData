# Input pipeline
# ref: http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/

import os, re
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import glob
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

data_folder = '/Users/wanfang/Downloads/hard_object_data/data_1'

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# read infomation from cvs files
data_ = []
for csvfile in sorted(glob.glob(data_folder+'/*.csv'), key=numericalSort):
    data_.append( pd.read_csv(csvfile) )

data = pd.concat(data_, ignore_index=True)

# read images, rotation angles, labels
all_images = ops.convert_to_tensor(sorted(glob.glob(data_folder+'/*.jpg'), key=numericalSort), dtype=dtypes.string)
all_angles= ops.convert_to_tensor(data.loc[1::2, 'Angle'].tolist(), dtype=dtypes.float32)
all_labels = ops.convert_to_tensor(data.loc[1::2, 'success'].tolist(), dtype=dtypes.int32)

def inputs(BATCH_SIZE = 128, validation = 0.2):

# create a partition vector
    N = len(data.loc[1::2, 'Angle'].tolist())
    test_set_size = int( (1-validation) * N )
    partitions = [0] * N
    partitions[:test_set_size] = [1] * test_set_size
    np.random.shuffle(partitions)

    # partition our data into a test and train set according to our partition vector
    train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
    train_angles, test_angles = tf.dynamic_partition(all_angles, partitions, 2)
    train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

    # create input queues
    train_input_queue = tf.train.slice_input_producer(
                                    [train_images, train_angles, train_labels],
                                    shuffle=False)
    test_input_queue = tf.train.slice_input_producer(
                                    [test_images, test_angles, test_labels],
                                    shuffle=False)

    # process path and string tensor into an image and a label
    IMAGE_HEIGHT = 200
    IMAGE_WIDTH = 200
    NUM_CHANNELS = 3
    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    train_angle = train_input_queue[1]
    train_label = train_input_queue[2]

    file_content = tf.read_file(test_input_queue[0])
    test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    test_angle = test_input_queue[1]
    test_label = test_input_queue[2]

    # define tensor shape
    train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


    # collect batches of images before processing
    train_image_batch, train_angle_batch, train_label_batch = tf.train.batch(
                                    [train_image, train_angle, train_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )
    test_image_batch, test_angle_batch, test_label_batch = tf.train.batch(
                                    [test_image, test_angle, test_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )
