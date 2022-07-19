####################
# Import Libraries
####################
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import glob

import argparse
import matplotlib.pyplot as plt
import os
import tensorflow_addons as tfa
import math
import imgaugtf
import segmentation_models as sm
import wandb
from tensorflow.keras.utils import get_custom_objects
import tensorflow_advanced_segmentation_models as tasm
import argparse

parser = argparse.ArgumentParser(description='train on tpu', add_help=True)

parser.add_argument('--run_wandb', action='store_true') # Default false
parser.add_argument('--use_bfloat16', action='store_false') # Default true
parser.add_argument('--exp_name', default='dev', type=str)
parser.add_argument('--exp_group', default='dev', type=str)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--warmup', default=0, type=int)
parser.add_argument('--image_size', default=384, type=int)
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--lr', default=0.002, type=float)
parser.add_argument('--backbone', default='efficientnetb2', type=str)
parser.add_argument('--arch', default='unet', type=str)
parser.add_argument('--val_fold', default=0, type=int)
parser.add_argument('--dataset_bucket', default='gs://uwmgi-tfrec-012-stride1', type=str)
parser.add_argument('--output_bucket', default='gs://train_outputs', type=str)
parser.add_argument('--tpu_name', default=None, type=str)
parser.add_argument('--verbose', default=2, type=int)
parser.add_argument('--pretrained', default='', type=str)
args = parser.parse_args()
print(args)

####################
# Config
####################
class CFG:
    vis_dataset=False
    run_wandb=args.run_wandb
    wandb_api_key=os.environ['WANDB_API_KEY']
    use_bfloat16=args.use_bfloat16
    exp_name=args.exp_name
    exp_group=args.exp_group
    epochs=args.epochs
    batch_size=args.batch_size
    lr = args.lr
    warmup = args.warmup
    image_size=args.image_size
    backbone=args.backbone
    val_fold=args.val_fold
    dataset_bucket = args.dataset_bucket
    output_bucket = args.output_bucket
    tpu_name = args.tpu_name
    arch = args.arch

    pretrained = args.pretrained
    if pretrained != '':
        target_file = sorted(tf.io.gfile.glob(os.path.join(pretrained,f'fold{val_fold}',f'fold{val_fold}','*')))[-1]
        if os.path.exists('model.h5'):
            os.remove('model.h5')
        tf.io.gfile.copy(target_file, 'model.h5')
        pretrained = 'model.h5'
    
####################
# Setup TPU
####################
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=CFG.tpu_name)
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
# Setup mixed bfloat16
####################
if CFG.use_bfloat16:
    policy = mixed_precision.Policy('mixed_bfloat16')
    mixed_precision.set_policy(policy)
    
####################
# Use wandb
####################
if CFG.run_wandb:
    wandb.login(key=CFG.wandb_api_key)
    cfg_dict = {k:v for k,v in dict(vars(CFG)).items() if '__' not in k}
    wandb.init(project="UWMGI", name=CFG.exp_name, entity="hirune924", group=CFG.exp_group, config=cfg_dict)

####################
# augmentation setup
####################
PIXEL_OPERATORS = [
    {"func": imgaugtf.seg.random_cutout, "option": {"num_holes": 8, "hole_size": 20, "replace": 0}},
    {"func": imgaugtf.seg.random_solarize, "option": {"threshold": 128}},
    {"func": imgaugtf.seg.random_solarize_add, "option": {"addition": 30, "threshold": 128}},
    {"func": imgaugtf.seg.random_color, "option": {"alpha_range": (0.2, 0.8)}},
    {"func": imgaugtf.seg.random_contrast, "option": {"lower": 0.2, "upper": 0.8}},
    {"func": imgaugtf.seg.random_brightness, "option": {"max_delta": 0.1}},
    {"func": imgaugtf.seg.random_posterize, "option": {"bits": 4}},
    {"func": imgaugtf.seg.random_invert, "option": {}},
    {"func": imgaugtf.seg.random_equalize, "option": {"bins": 256}},
    {"func": imgaugtf.seg.random_sharpness, "option": {"alpha_range": (-3.0, 3.0)}},
    {"func": imgaugtf.seg.random_autocontrast, "option": {}},
    {"func": imgaugtf.seg.random_hsv_in_yiq, "option": {"max_delta_hue": 0.2, "lower_saturation": 0.5, "upper_saturation": 1.0, "lower_value": 0.5, "upper_value": 1.0}},
    {"func": imgaugtf.seg.random_gaussian_filter2d, "option": {"filter_shape": (3, 3)}},
    {"func": imgaugtf.seg.random_mean_filter2d, "option": {"filter_shape": (3, 3)}},
    {"func": imgaugtf.seg.random_median_filter2d, "option": {"filter_shape": (3, 3)}},
    {"func": imgaugtf.seg.random_gray, "option": {}},
    {"func": imgaugtf.seg.random_hue, "option": {"max_delta": 0.2}},
    {"func": imgaugtf.seg.random_saturation, "option": {"saturation_factor": (0.75, 1.25)}},
    {"func": imgaugtf.seg.random_gamma, "option": {"gamma_range": (0.75, 1.25)}},
    {"func": imgaugtf.seg.random_jpeg_quality, "option": {"jpeg_quality_range": (75, 95)}},
    {"func": imgaugtf.seg.random_gaussian_noise, "option": {"stddev_range": (5, 95)}},
    {"func": imgaugtf.seg.random_speckle_noise, "option": {"prob_range": (0.0, 0.05)}},
]
GEO_OPERATORS = [
    {"func": imgaugtf.seg.random_flip_left_right, "option": {}},
    {"func": imgaugtf.seg.random_flip_up_down, "option": {}},
    {"func": imgaugtf.seg.random_rotate, "option": {"degree_range": (-90, 90), "interpolation": "nearest", "fill_mode": "constant", "fill_value": 0.0}},
    #{"func": imgaugtf.seg.random_translate_x, "option": {"percent": 0.5, 'interpolation': 'nearest', 'fill_mode': 'constant', 'fill_value': 0.0}},
    #{"func": imgaugtf.seg.random_translate_y, "option": {"percent": 0.5, 'interpolation': 'nearest', 'fill_mode': 'constant', 'fill_value': 0.0}},
    #{"func": imgaugtf.seg.random_shear_x, "option": {"percent": 0.3, 'interpolation': 'nearest', 'fill_mode': 'constant', 'fill_value': 0.0}},
    #{"func": imgaugtf.seg.random_shear_y, "option": {"percent": 0.3, 'interpolation': 'nearest', 'fill_mode': 'constant', 'fill_value': 0.0}},
    #{"func": imgaugtf.seg.random_zoom, "option": {"scale": (-0.2, 0.2), "interpolation": "nearest", "fill_mode": "constant", "fill_value": 0}},
    #{"func": imgaugtf.seg.random_grid_shuffle, "option": {"grid_size": (3, 3)}},
    {"func": imgaugtf.seg.random_affine, "option": {"translate": (-0.3, 0.3), "shear": (-0.3, 0.3), "rotate": (-90, 90), "scale": (0.75, 1.25)
    , "interpolation": 'nearest', "fill_mode": 'constant', "fill_value": 0}},
    {"func": imgaugtf.seg.random_elastic_deform, "option": {"scale": 10, "strength": 10}},
    {"func": imgaugtf.seg.random_sparse_warp, "option": {"dst_x": 0.15, "dst_y": 0.15}},
]

####################
# Dataset functions
####################
def read_tfrecord(example):
    tfrec_format = {
        #"image_id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        'mask': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    example['image'] = tf.reshape(tf.io.decode_raw(example['image'], tf.float32), (example['height'], example['width'], 3))
    example['mask'] = tf.reshape(tf.io.decode_raw(example['mask'], tf.int64), (example['height'], example['width'], 3))
    
    example['image'] = tf.cast(example['image'], tf.uint8)
    example['mask'] = tf.cast(example['mask'], tf.uint8) * 255
    return example

def prepocess(example):
    image = example['image']
    mask = example['mask']
    image, mask = imgaugtf.seg.random_resized_crop(image, mask, size=[CFG.image_size, CFG.image_size], prob=0.0)
    image.set_shape([CFG.image_size, CFG.image_size, 3])
    mask.set_shape([CFG.image_size, CFG.image_size, 3])
    image = tf.cast(image, tf.float32)/ 255.0
    image = image * 2.0 - 1.0
    mask = tf.cast(mask, tf.float32)/255.0
    return image, mask

def prepocess_with_aug(example):
    image = example['image']
    mask = example['mask']
    image, mask = imgaugtf.seg.random_resized_crop(image, mask, size=[CFG.image_size, CFG.image_size], prob=0.0)
    image.set_shape([CFG.image_size, CFG.image_size, 3])
    mask.set_shape([CFG.image_size, CFG.image_size, 3])
    image, mask = imgaugtf.seg.apply_n(image, mask, functions=PIXEL_OPERATORS, num_ops=2, prob=1.0)
    image, mask = imgaugtf.seg.apply_n(image, mask, functions=GEO_OPERATORS, num_ops=2, prob=1.0)
    
    image.set_shape([CFG.image_size, CFG.image_size, 3])
    mask.set_shape([CFG.image_size, CFG.image_size, 3])
    image = tf.cast(image, tf.float32)/255.0
    image = image * 2.0 - 1.0
    mask = tf.cast(mask, tf.float32)/255.0
    return image, mask

####################
# Dataset
####################
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = CFG.batch_size * strategy.num_replicas_in_sync

train_tfrec = []
valid_tfrec = []
for i in [i for i in range(10) if i //2 != CFG.val_fold]:
    train_tfrec += tf.io.gfile.glob(os.path.join(CFG.dataset_bucket, f'uwmgi_train_fold{i}_*.tfrec'))
for i in [i for i in range(10) if i //2 == CFG.val_fold]:
    valid_tfrec += tf.io.gfile.glob(os.path.join(CFG.dataset_bucket, f'uwmgi_train_fold{i}_*.tfrec'))

train_dataset = tf.data.TFRecordDataset(train_tfrec, num_parallel_reads=AUTO, compression_type="GZIP")
num_train_samples = train_dataset.reduce(0,lambda x,_:x + 1).numpy()
train_dataset = (train_dataset
.map(read_tfrecord)
.cache()
.repeat()
.shuffle(2048, reshuffle_each_iteration=True)
.map(prepocess_with_aug, num_parallel_calls=AUTO)
.batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
.prefetch(AUTO))

valid_dataset = tf.data.TFRecordDataset(valid_tfrec, num_parallel_reads=AUTO, compression_type="GZIP")
num_valid_samples = valid_dataset.reduce(0,lambda x,_:x + 1).numpy()
valid_dataset = (valid_dataset
.map(read_tfrecord)
.map(prepocess)
.batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
#.cache()
.prefetch(AUTO))

num_train_steps = num_train_samples//BATCH_SIZE
num_valid_steps = num_valid_samples//BATCH_SIZE
print(f'num_train_steps: {num_train_steps}, num_valid_steps: {num_valid_steps}')

####################
# Visualize dataset
####################
if CFG.vis_dataset:
    def show_images(images, cols=1, titles=None):
        """Display a list of images in a single figure with matplotlib.
        
        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.
        
        cols (Default = 1): Number of columns in figure (number of rows is 
                            set to np.ceil(n_images/float(cols))).
        
        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """
        assert((titles is None)or (len(images) == len(titles)))
        n_images = len(images)
        if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, int(np.ceil(n_images/float(cols))), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title, fontsize=15*int(np.ceil(n_images/float(cols))))
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()

    images = []
    for img, mask in train_dataset.take(30):
        img = np.clip((((img[0].numpy()/2.0 +0.5)*255).astype(np.int32) + mask[0].numpy().astype(np.int32)*255),0,255).astype(np.uint8)
        images.append(img)
    show_images(images, cols=5)
    
def dice_coe(output, target, axis = [1,2], smooth=1e-10):
    output = tf.dtypes.cast( tf.math.greater(output, 0.5), tf. float32 )
    target = tf.dtypes.cast( tf.math.greater(target, 0.5), tf. float32 )
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)

    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice, axis=1)
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice

####################
# Build model
####################
with strategy.scope():
    sm.set_framework('tf.keras')

    sm.framework()
    if CFG.arch == 'unet':
        model = sm.Unet(CFG.backbone, input_shape=(CFG.image_size, CFG.image_size, 3), classes=3, activation='sigmoid', encoder_weights='imagenet')
    elif CFG.arch == 'fpn':
        model = sm.FPN(CFG.backbone, input_shape=(CFG.image_size, CFG.image_size, 3), classes=3, activation='sigmoid', encoder_weights='imagenet')
    elif CFG.arch == 'linknet':
        model = sm.Linknet(CFG.backbone, input_shape=(CFG.image_size, CFG.image_size, 3), classes=3, activation='sigmoid', encoder_weights='imagenet')
    elif CFG.arch == 'pspnet':
        model = sm.PSPNet(CFG.backbone, input_shape=(CFG.image_size, CFG.image_size, 3), classes=3, activation='sigmoid', encoder_weights='imagenet')

    if CFG.pretrained != '':
        model.load_weights(CFG.pretrained)
    model.compile(optimizer='adam',
                  #loss=sm.losses.binary_crossentropy,
                  #loss=tf.keras.losses.BinaryCrossentropy(),
                  #metrics=[dice_coe]
                  loss=sm.losses.BinaryCELoss()+sm.losses.DiceLoss(beta=1, per_image=False),
                  metrics=[sm.metrics.IOUScore(threshold=0.5, per_image=True), sm.metrics.FScore(threshold=0.5, per_image=True), dice_coe,
                  sm.metrics.Precision(threshold=0.5, per_image=True), sm.metrics.Recall(threshold=0.5, per_image=True)],
                  steps_per_execution=1,
                  #jit_compile=True
                 )
    #model.summary()
 
####################
# Train
####################
import math

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

callbacks = []
callbacks.append(get_cosine_schedule_with_warmup(lr=CFG.lr, num_warmup_steps=CFG.warmup, num_training_steps=CFG.epochs))
callbacks.append(tf.keras.callbacks.ModelCheckpoint(CFG.exp_name + '-fold'+str(CFG.val_fold) + '-epoch{epoch:02d}-dice{val_f1-score:.4f}.h5', monitor='val_f1-score', verbose=0, save_best_only=True, save_weights_only=False, mode='max', save_freq='epoch'))
callbacks.append(tf.keras.callbacks.CSVLogger('training.log'))
if CFG.run_wandb:
    callbacks.append(wandb.keras.WandbCallback(save_model=False))
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=CFG.epochs,
    callbacks=callbacks,
    steps_per_epoch=num_train_steps,
    validation_steps=num_valid_steps,
    verbose=args.verbose
)