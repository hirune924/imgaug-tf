import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import math
from . import functional as F

__all__ = [
    'random_flip_left_right',
]

def apply_func_with_prob(func, image, mask, args, prob):
    """Apply `func` to image w/ `args` as input with probability `prob`."""

    # Apply the function with probability `prob`.
    should_apply_op = tf.cast(
        tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)

    augmented_image = tf.cond(
        should_apply_op,
        lambda: func(image, *args),
        lambda: image)
    augmented_mask = tf.cond(
        should_apply_op,
        lambda: func(mask, *args),
        lambda: mask)
    return augmented_image, augmented_mask

def random_flip_left_right(image, mask, prob=0.5):
    return apply_func_with_prob(tf.image.flip_left_right, image, mask, (), prob)