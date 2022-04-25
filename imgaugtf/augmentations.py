import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import math
from functions import *

def apply_func_with_prob(func, image, args, prob):
    """Apply `func` to image w/ `args` as input with probability `prob`."""

    # Apply the function with probability `prob`.
    should_apply_op = tf.cast(
        tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
    augmented_image = tf.cond(
        should_apply_op,
        lambda: func(image, *args),
        lambda: image)
    return augmented_image

def random_solarize(image, threshold=128, prob=0.5):

    return apply_func_with_prob(solarize, image, (threshold, ), prob)

def random_solarize_add(image, addition=30, threshold=128, prob=0.5):

    return apply_func_with_prob(solarize_add, image, (addition, threshold, ), prob)

def random_color(image, alpha_range=(0.2, 0.8), prob=0.5):
    '''if alpha is 0, return gray image. alpha is 1, do nothing.'''
    alpha = tf.random.uniform([], minval=alpha_range[0], maxval=alpha_range[1], dtype=tf.float32)
    return apply_func_with_prob(color, image, (alpha, ), prob)

def random_contrast(image, lower=0.2, upper=0.8, seed=None, prob=0.5):
    return apply_func_with_prob(tf.image.random_contrast, image, (lower, upper, seed,), prob)

def random_brightness(image, max_delta=0.1, seed=None, prob=0.5):
    return apply_func_with_prob(tf.image.random_brightness, image, (max_delta, seed,), prob)

def random_posterize(image, bits=4, prob=0.5):
    return apply_func_with_prob(posterize, image, (bits,), prob)

def random_rotate(image, degree_range=(-90, 90), replace=0, prob=0.5):
    degree = tf.random.uniform([], minval=degree_range[0], maxval=degree_range[1], dtype=tf.float32)
    return apply_func_with_prob(rotate, image, (degree, replace, ), prob)

def random_invert(image, prob=0.5):
    return apply_func_with_prob(invert, image, (), prob)

def random_gray(image, prob=0.5):
    '''return 1 channel image. this should be reimplement. recommend use random_color!!'''
    return apply_func_with_prob(tf.image.rgb_to_grayscale, image, (), prob)

def random_equalize(image, bins=256, prob=0.5):
    #return apply_func_with_prob(tfa.image.equalize, image, (bins,), prob)
    return apply_func_with_prob(equalize, image, (), prob)

def random_sharpness(image, alpha_range=(-3.0, 3.0), prob=0.5):
    alpha = tf.random.uniform([], minval=alpha_range[0], maxval=alpha_range[1], dtype=tf.float32)
    return apply_func_with_prob(sharpness, image, (alpha, ), prob)

def random_autocontrast(image, prob=0.5):
    return apply_func_with_prob(autocontrast, image, (), prob)

def random_translate_x(image, percent=0.5, replace=0, prob=0.5):
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    pixels = tf.cast(image_width, tf.float32) * tf.random.uniform([], minval=-percent, maxval=percent, dtype=tf.float32)
    return apply_func_with_prob(translate_x, image, (pixels, replace), prob)

def random_translate_y(image, percent=0.5, replace=0, prob=0.5):
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    pixels = tf.cast(image_height, tf.float32) * tf.random.uniform([], minval=-percent, maxval=percent, dtype=tf.float32)
    return apply_func_with_prob(translate_y, image, (pixels, replace), prob)

def random_shear_x(image, percent=0.3, replace=0, prob=0.5):
    level = tf.random.uniform([], minval=-percent, maxval=percent, dtype=tf.float32)
    return apply_func_with_prob(shear_x, image, (level, replace), prob)

def random_shear_y(image, percent=0.3, replace=0, prob=0.5):
    level = tf.random.uniform([], minval=-percent, maxval=percent, dtype=tf.float32)
    return apply_func_with_prob(shear_y, image, (level, replace), prob)

def random_hsv_in_yiq(image, max_delta_hue=0.2, lower_saturation=0.5, upper_saturation=1., lower_value=0.5, upper_value=1.0, prob=0.5):
    return apply_func_with_prob(tfa.image.random_hsv_in_yiq, image, (max_delta_hue, lower_saturation, upper_saturation, lower_value, upper_value), prob)

def random_gaussian_filter2d(image, filter_shape_range=(3,7), prob=0.5):
    ksize = tf.random.uniform([], minval=filter_shape_range[0], maxval=filter_shape_range[1], dtype=tf.int32)
    #sigma = 0.3*((tf.cast(ksize, tf.float32)-1)*0.5 - 1) + 0.8
    return apply_func_with_prob(tfa.image.gaussian_filter2d, image, ([ksize, ksize], 1.0), prob)

def random_mean_filter2d(image, filter_shape=(3,3), prob=0.5):
    #ksize = tf.random.uniform([], minval=filter_shape_range[0], maxval=filter_shape_range[1], dtype=tf.int32)
    return apply_func_with_prob(tfa.image.mean_filter2d, image, (filter_shape,), prob)

def random_median_filter2d(image, filter_shape=(3,3), prob=0.5):
    #ksize = tf.random.uniform([], minval=filter_shape_range[0], maxval=filter_shape_range[1], dtype=tf.int32)
    return apply_func_with_prob(tfa.image.median_filter2d, image, (filter_shape,), prob)

def random_resized_crop(image, size=[256, 256], area_range=(0.05, 1.0), aspect_ratio_range=(0.75,1.33), prob=1.0):
    def _random_resized_crop(image, size, area_range=(0.05, 1.0), aspect_ratio_range=(0.75,1.33)):
        image = random_crop(image, area_range=area_range, aspect_ratio_range=aspect_ratio_range)
        image = tf.image.resize(image, size=size)
        image = tf.cast(image, tf.uint8)
        return image
    #return apply_func_with_prob(_random_resized_crop, image, (size, area_range, aspect_ratio_range), prob)
    #return _random_resized_crop(image, size, area_range=(0.05, 1.0), aspect_ratio_range=(0.75,1.33))
    return tf.cond(tf.random.uniform([], 0, 1) < prob,
     lambda: _random_resized_crop(image, size, area_range=area_range, aspect_ratio_range=aspect_ratio_range),
     lambda: tf.cast(tf.image.resize(image, size=size), tf.uint8))