####################
# Import Libraries
####################
import numpy as np
import tensorflow as tf
import glob

import argparse
import matplotlib.pyplot as plt
import os
import tensorflow_addons as tfa
import math
import imgaugtf

####################
# Setup TPU
####################
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



####################
# Dataset functions
####################
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    return image

def read_tfrecord(example):
    tfrec_format = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        'label': tf.io.FixedLenFeature([], tf.int64),
        'breed': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    example['image'] = tf.image.decode_jpeg(example['image'], channels=3)
    return example


def prepocess(example):
    image = example['image']
    image = imgaugtf.random_resized_crop(image, size=[256, 256], prob=0.0)
    image.set_shape([256, 256, 3])
    image = tf.cast(image, tf.float32)# / 255.0
    return image, example['label']

def prepocess_with_aug(example):
    image = example['image']
    image = imgaugtf.random_resized_crop(image, size=[256, 256], prob=1.0)
    image.set_shape([256, 256, 3])
    image = imgaugtf.apply_n(image, functions=imgaugtf.PIXEL_OPERATORS, num_ops=2, prob=1.0)
    #image.set_shape([256, 256, 3])
    image = imgaugtf.apply_n(image, functions=imgaugtf.GEO_OPERATORS, num_ops=2, prob=1.0)
    
    image.set_shape([256, 256, 3])
    image = tf.cast(image, tf.float32)
    return image, example['label']

GCS_DS_PATH = KaggleDatasets().get_gcs_path('dogbreedtfrec')

####################
# Dataset
####################
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

train_tfrec = []
valid_tfrec = []
for i in [0,1,2,3]:
    train_tfrec += tf.io.gfile.glob(os.path.join(GCS_DS_PATH, f'dogbreed_train_fold{i}_*.tfrec'))
valid_tfrec += tf.io.gfile.glob(os.path.join(GCS_DS_PATH, f'dogbreed_train_fold4_*.tfrec'))

train_dataset = (tf.data.TFRecordDataset(train_tfrec, num_parallel_reads=AUTO, compression_type="GZIP")
.map(read_tfrecord)
.cache()
.shuffle(2048)
.repeat()
.prefetch(AUTO)
.map(prepocess_with_aug, num_parallel_calls=AUTO)
.batch(BATCH_SIZE))

valid_dataset = (tf.data.TFRecordDataset(valid_tfrec, num_parallel_reads=AUTO, compression_type="GZIP")
.map(read_tfrecord)
.map(prepocess)
.batch(BATCH_SIZE)
#.cache()
.prefetch(AUTO))

with strategy.scope():
    model_base = tf.keras.applications.efficientnet.EfficientNetB7(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(256, 256, 3),
        pooling='avg',
        #classes=120,
        #classifier_activation='softmax'
    )
    model = tf.keras.Sequential([
        model_base,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(120, activation="softmax")
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.summary()
 
####################
# Train
####################
import math
LR = 0.0002 # 0.0005
EPOCHS = 50
WARMUP = 4

def get_cosine_schedule_with_warmup(lr, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Modified the get_cosine_schedule_with_warmup from huggingface for tensorflow
    (https://huggingface.co/transformers/_modules/transformers/optimization.html#get_cosine_schedule_with_warmup)

    Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return float(epoch) / float(max(1, num_warmup_steps)) * lr
        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)


lr_schedule= get_cosine_schedule_with_warmup(lr=LR, num_warmup_steps=WARMUP, num_training_steps=EPOCHS)

history = model.fit(
  train_dataset,
  validation_data=valid_dataset,
  steps_per_epoch=64,
  epochs=EPOCHS,
  callbacks=[lr_schedule]
)