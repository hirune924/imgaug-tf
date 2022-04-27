import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import math
from . import functional as F

__all__ = [
    "random_flip_left_right",
]


def apply_func_with_prob_mask(func, image, mask, args, mask_args, prob):
    """Apply `func` to image w/ `args` as input with probability `prob`."""

    # Apply the function with probability `prob`.
    should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)

    augmented_image = tf.cond(should_apply_op, lambda: func(image, *args), lambda: image)
    augmented_mask = tf.cond(should_apply_op, lambda: func(mask, *mask_args), lambda: mask)
    return augmented_image, augmented_mask


def apply_func_with_prob(func, image, args, prob):
    """Apply `func` to image w/ `args` as input with probability `prob`."""

    # Apply the function with probability `prob`.
    should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)

    augmented_image = tf.cond(should_apply_op, lambda: func(image, *args), lambda: image)
    return augmented_image


def random_flip_left_right(image, mask, prob=0.5):
    return apply_func_with_prob_mask(tf.image.flip_left_right, image, mask, (), (), prob)


def random_flip_up_down(image, mask, prob=0.5):
    return apply_func_with_prob_mask(tf.image.flip_up_down, image, mask, (), (), prob)


def random_solarize(image, mask, threshold=128, prob=0.5):
    return apply_func_with_prob(F.solarize, image, (threshold,), prob), mask


def random_solarize_add(image, mask, addition=30, threshold=128, prob=0.5):

    return (
        apply_func_with_prob(
            F.solarize_add,
            image,
            (
                addition,
                threshold,
            ),
            prob,
        ),
        mask,
    )


def random_color(image, mask, alpha_range=(0.2, 0.8), prob=0.5):
    """if alpha is 0, return gray image. alpha is 1, do nothing."""
    alpha = tf.random.uniform([], minval=alpha_range[0], maxval=alpha_range[1], dtype=tf.float32)
    return apply_func_with_prob(F.color, image, (alpha,), prob), mask


def random_contrast(image, mask, lower=0.2, upper=0.8, seed=None, prob=0.5):
    return (
        apply_func_with_prob(
            tf.image.random_contrast,
            image,
            (
                lower,
                upper,
                seed,
            ),
            prob,
        ),
        mask,
    )


def random_brightness(image, mask, max_delta=0.1, seed=None, prob=0.5):
    return (
        apply_func_with_prob(
            tf.image.random_brightness,
            image,
            (
                max_delta,
                seed,
            ),
            prob,
        ),
        mask,
    )


def random_posterize(image, mask, bits=4, prob=0.5):
    return apply_func_with_prob(F.posterize, image, (bits,), prob), mask


def random_rotate(image, mask, degree_range=(-90, 90), interpolation="nearest", fill_mode="constant", fill_value=0.0, prob=0.5):
    degree = tf.random.uniform([], minval=degree_range[0], maxval=degree_range[1], dtype=tf.float32)
    return apply_func_with_prob_mask(
        F.rotate,
        image,
        mask,
        (
            degree,
            interpolation,
            fill_mode,
            fill_value,
        ),
        (
            degree,
            "nearest",
            fill_mode,
            fill_value,
        ),
        prob,
    )
