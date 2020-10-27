# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


features={'uid':tf.io.FixedLenFeature([1],tf.int64),
            'age':tf.io.FixedLenFeature([1],dtype=tf.int64),
            'sex': tf.io.FixedLenFeature([1],dtype=tf.int64),
            'city_type': tf.io.FixedLenFeature([1],dtype=tf.int64),
            'masked_positions_res': tf.io.VarLenFeature(dtype=tf.int64),
            'masked_gid_list': tf.io.VarLenFeature(dtype=tf.string),
            'masked_gid_labels': tf.io.VarLenFeature(dtype=tf.string),
            'masked_cid3_list':tf.io.VarLenFeature(tf.string),
            'masked_cid3_labels':tf.io.VarLenFeature(dtype=tf.string),
            'masked_merchant_list': tf.io.VarLenFeature(dtype=tf.string),
            'masked_merchant_labels': tf.io.VarLenFeature(dtype=tf.string),
            'masked_brand_list': tf.io.VarLenFeature(dtype=tf.string),
            'masked_brand_labels': tf.io.VarLenFeature(dtype=tf.string),
            'duration_tokens': tf.io.VarLenFeature(dtype=tf.string)
        }




def parse_exmp(serial_exmp):
    feats = tf.io.parse_single_example(serial_exmp, features)
    return feats


def train_input_fn(filenames,shuffle_buffer_size):
    dataset = (tf.data.Dataset.from_tensor_slices(filenames)
               .interleave(lambda fn: tf.data.TFRecordDataset(fn),
                           cycle_length=4,
                           block_length=4))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_exmp, num_parallel_calls=8)

    for elem in dataset:
        print(elem["duration_tokens"])


if __name__=='__main__':
    train_files=["filepath"]
    train_input_fn(train_files,1000)
