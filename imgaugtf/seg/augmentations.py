import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import math
from .. import functional as F


def apply_func_with_prob_mask(func, mask_func, image, mask, args, mask_args, prob):
    """Apply `func` to image w/ `args` as input with probability `prob`."""

    # Apply the function with probability `prob`.
    should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)

    augmented_image = tf.cond(should_apply_op, lambda: func(image, *args), lambda: image)
    augmented_mask = tf.cond(should_apply_op, lambda: mask_func(mask, *mask_args), lambda: mask)
    return augmented_image, augmented_mask


def apply_func_with_prob(func, image, args, prob):
    """Apply `func` to image w/ `args` as input with probability `prob`."""

    # Apply the function with probability `prob`.
    should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)

    augmented_image = tf.cond(should_apply_op, lambda: func(image, *args), lambda: image)
    return augmented_image


def random_flip_left_right(image, mask, prob=0.5):
    return apply_func_with_prob_mask(tf.image.flip_left_right, tf.image.flip_left_right, image, mask, (), (), prob)


def random_flip_up_down(image, mask, prob=0.5):
    return apply_func_with_prob_mask(tf.image.flip_up_down, tf.image.flip_up_down, image, mask, (), (), prob)


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

def random_invert(image, mask, prob=0.5):
    return apply_func_with_prob(F.invert, image, (), prob), mask


def random_gray(image, mask, prob=0.5):
    """return 1 channel image. this should be reimplement. recommend use random_color!!"""
    def _to_gray(image):
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return apply_func_with_prob(_to_gray, image, (), prob), mask


def random_equalize(image, mask, bins=256, prob=0.5):
    # return apply_func_with_prob(tfa.image.equalize, image, (bins,), prob)
    return apply_func_with_prob(F.equalize, image, (), prob), mask


def random_sharpness(image, mask, alpha_range=(-3.0, 3.0), prob=0.5):
    alpha = tf.random.uniform([], minval=alpha_range[0], maxval=alpha_range[1], dtype=tf.float32)
    return apply_func_with_prob(F.sharpness, image, (alpha,), prob), mask


def random_autocontrast(image, mask, prob=0.5):
    return apply_func_with_prob(F.autocontrast, image, (), prob), mask



def random_translate_x(image, mask, percent=0.5, interpolation='nearest', fill_mode='constant', fill_value=0.0, prob=0.5):
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    pixels = tf.cast(image_width, tf.float32) * tf.random.uniform([], minval=-percent, maxval=percent, dtype=tf.float32)
    #return apply_func_with_prob(F.translate_x, image, (pixels, interpolation, fill_mode, fill_value), prob)
    return apply_func_with_prob_mask(
        F.translate_x,
        F.translate_x,
        image,
        mask,
        (
            pixels,
            interpolation,
            fill_mode,
            fill_value,
        ),
        (
            pixels,
            "nearest",
            fill_mode,
            fill_value,
        ),
        prob,
    )

def random_translate_y(image, mask, percent=0.5, interpolation='nearest', fill_mode='constant', fill_value=0.0, prob=0.5):
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    pixels = tf.cast(image_height, tf.float32) * tf.random.uniform([], minval=-percent, maxval=percent, dtype=tf.float32)
    #return apply_func_with_prob(F.translate_y, image, (pixels, interpolation, fill_mode, fill_value), prob)
    return apply_func_with_prob_mask(
        F.translate_y,
        F.translate_y,
        image,
        mask,
        (
            pixels,
            interpolation,
            fill_mode,
            fill_value,
        ),
        (
            pixels,
            "nearest",
            fill_mode,
            fill_value,
        ),
        prob,
    )

def random_shear_x(image, mask, percent=0.3, interpolation='nearest', fill_mode='constant', fill_value=0.0, prob=0.5):
    level = tf.random.uniform([], minval=-percent, maxval=percent, dtype=tf.float32)
    #return apply_func_with_prob(F.shear_x, image, (level, interpolation, fill_mode, fill_value), prob)
    return apply_func_with_prob_mask(
        F.shear_x,
        F.shear_x,
        image,
        mask,
        (
            level,
            interpolation,
            fill_mode,
            fill_value,
        ),
        (
            level,
            "nearest",
            fill_mode,
            fill_value,
        ),
        prob,
    )

def random_shear_y(image, mask, percent=0.3, interpolation='nearest', fill_mode='constant', fill_value=0.0, prob=0.5):
    level = tf.random.uniform([], minval=-percent, maxval=percent, dtype=tf.float32)
    #return apply_func_with_prob(F.shear_y, image, (level, interpolation, fill_mode, fill_value), prob)
    return apply_func_with_prob_mask(
        F.shear_y,
        F.shear_y,
        image,
        mask,
        (
            level,
            interpolation,
            fill_mode,
            fill_value,
        ),
        (
            level,
            "nearest",
            fill_mode,
            fill_value,
        ),
        prob,
    )

def random_hsv_in_yiq(image, mask, max_delta_hue=0.2, lower_saturation=0.5, upper_saturation=1.0, lower_value=0.5, upper_value=1.0, prob=0.5):
    return apply_func_with_prob(tfa.image.random_hsv_in_yiq, image, (max_delta_hue, lower_saturation, upper_saturation, lower_value, upper_value), prob), mask


def random_gaussian_filter2d(image, mask, filter_shape=(3, 3), prob=0.5):
    #ksize = tf.random.uniform([], minval=filter_shape_range[0], maxval=filter_shape_range[1], dtype=tf.int32), mask
    # sigma = 0.3*((tf.cast(ksize, tf.float32)-1)*0.5 - 1) + 0.8
    return apply_func_with_prob(tfa.image.gaussian_filter2d, image, (filter_shape, 1.0), prob), mask


def random_mean_filter2d(image, mask, filter_shape=(3, 3), prob=0.5):
    # ksize = tf.random.uniform([], minval=filter_shape_range[0], maxval=filter_shape_range[1], dtype=tf.int32)
    return apply_func_with_prob(tfa.image.mean_filter2d, image, (filter_shape,), prob), mask


def random_median_filter2d(image, mask, filter_shape=(3, 3), prob=0.5):
    # ksize = tf.random.uniform([], minval=filter_shape_range[0], maxval=filter_shape_range[1], dtype=tf.int32)
    return apply_func_with_prob(tfa.image.median_filter2d, image, (filter_shape,), prob), mask


def random_bbox_crop(image, mask, area_range=(0.05, 1.0), aspect_ratio_range=(0.75, 1.33)):
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), tf.zeros([0, 0, 4], tf.float32), 
        area_range=area_range, 
        aspect_ratio_range=aspect_ratio_range, 
        min_object_covered=0, 
        use_image_if_no_bounding_boxes=True, 
        seed=0
    )
    image = tf.slice(image, begin, size)
    mask = tf.slice(mask, begin, size)
    image.set_shape([None, None, 3])
    mask.set_shape([None, None, None])
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, dtype=tf.uint8)
    mask = tf.cast(mask, dtype=tf.uint8)
    return image, mask


def random_resized_crop(image, mask, size=[256, 256], area_range=(0.05, 1.0), aspect_ratio_range=(0.75, 1.33), prob=1.0):
    def _random_resized_crop(image, mask, size, area_range=(0.05, 1.0), aspect_ratio_range=(0.75, 1.33)):
        image, mask = random_bbox_crop(image, mask, area_range=area_range, aspect_ratio_range=aspect_ratio_range)
        image = tf.image.resize(image, size=size)
        mask = tf.image.resize(mask, size=size)
        image = tf.cast(image, tf.uint8)
        mask = tf.cast(mask, dtype=tf.uint8)
        return image, mask

    # return apply_func_with_prob(_random_resized_crop, image, (size, area_range, aspect_ratio_range), prob)
    # return _random_resized_crop(image, size, area_range=(0.05, 1.0), aspect_ratio_range=(0.75,1.33))
    return tf.cond(
        tf.random.uniform([], 0, 1) < prob,
        lambda: _random_resized_crop(image, mask, size, area_range=area_range, aspect_ratio_range=aspect_ratio_range),
        lambda: (tf.cast(tf.image.resize(image, size=size), tf.uint8), tf.cast(tf.image.resize(mask, size=size), tf.uint8)),
    )

def random_cutout(image, mask, num_holes=8, hole_size=20, replace=0, prob=0.5):
    def _random_cutout(image, num_holes, hole_size, replace):
        for _ in range(num_holes):
            image_height = tf.shape(image)[0]
            image_width = tf.shape(image)[1]

            # Sample the center location in the image where the zero mask will be applied.
            cutout_center_height = tf.random.uniform(shape=[], minval=0, maxval=image_height, dtype=tf.int32)
            cutout_center_width = tf.random.uniform(shape=[], minval=0, maxval=image_width, dtype=tf.int32)
            
            image = F.cutout(image, pad_size=hole_size // 2, cutout_center_height=cutout_center_height, cutout_center_width=cutout_center_width, replace=replace)
        return image

    return apply_func_with_prob(_random_cutout, image, (num_holes, hole_size, replace), prob), mask


def random_zoom(image, mask, scale=(0.2, 0.2), interpolation: str = "nearest", fill_mode="constant", fill_value=0, prob=0.5):
    scale = tf.random.uniform([], 1.0 - scale[0], 1.0 + scale[0])
    # scale_y = tf.random.uniform([], 1.0 - scale[1], 1.0 + scale[1])
    #return apply_func_with_prob(F.scale_xy, image, ((scale, scale), interpolation, fill_mode, fill_value), prob)
    return apply_func_with_prob_mask(
        F.scale_xy,
        F.scale_xy,
        image,
        mask,
        (
            (scale, scale),
            interpolation,
            fill_mode,
            fill_value,
        ),
        (
            (scale, scale),
            "nearest",
            fill_mode,
            fill_value,
        ),
        prob,
    )

def random_grid_shuffle(image, mask, grid_size=(3, 3), prob=0.5):
    size = tf.shape(image)
    #grid_x = size[1] // grid_size[0]
    grid_x = tf.math.floordiv(size[1], grid_size[0])
    #grid_y = size[0] // grid_size[1]
    grid_y = tf.math.floordiv(size[0], grid_size[1])
    order = tf.random.shuffle([ i for i in range(grid_size[0] * grid_size[1]) ])
    #return apply_func_with_prob(F.grid_shuffle, image, (grid_x, grid_y, grid_size, order), prob)
    return apply_func_with_prob_mask(
        F.grid_shuffle,
        F.grid_shuffle,
        image,
        mask,
        (
            grid_x,
            grid_y,
            grid_size,
            order,
        ),
        (
            grid_x,
            grid_y,
            grid_size,
            order,
        ),
        prob,
    )

def random_affine(image, mask, translate=(-0.3, 0.3), shear=(-0.3, 0.3), rotate=(-90, 90), scale=(0.75, 1.25), interpolation='nearest', fill_mode='constant', fill_value=0, prob=0.5):
    size = tf.shape(image)
    trans_x = tf.random.uniform([], minval=translate[0], maxval=translate[1], dtype=tf.float32) * tf.cast(size[1], tf.float32)
    trans_y = tf.random.uniform([], minval=translate[0], maxval=translate[1], dtype=tf.float32) * tf.cast(size[0], tf.float32)
    shear_x = tf.random.uniform([], minval=shear[0], maxval=shear[1], dtype=tf.float32)
    shear_y = tf.random.uniform([], minval=shear[0], maxval=shear[1], dtype=tf.float32)
    scale_x = tf.random.uniform([], minval=scale[0], maxval=scale[1], dtype=tf.float32)
    scale_y = tf.random.uniform([], minval=scale[0], maxval=scale[1], dtype=tf.float32)    
    degree = tf.random.uniform([], minval=rotate[0], maxval=rotate[1], dtype=tf.float32)
    #return apply_func_with_prob(F.affine, image, (trans_x, trans_y, shear_x, shear_y, scale_x, scale_y, degree, interpolation, fill_mode, fill_value), prob)
    return apply_func_with_prob_mask(
        F.affine,
        F.affine,
        image,
        mask,
        (trans_x, trans_y, shear_x, shear_y, scale_x, scale_y, degree, interpolation, fill_mode, fill_value),
        (trans_x, trans_y, shear_x, shear_y, scale_x, scale_y, degree, "nearest", fill_mode, fill_value),
        prob,
    )



def random_hue(image, mask, max_delta=0.2, prob=0.5):
    delta = tf.random.uniform([], minval=-max_delta, maxval=max_delta, dtype=tf.float32)
    return apply_func_with_prob(F.adjust_hue, image, (delta, ), prob), mask


def random_saturation(image, mask, saturation_factor=(0.75, 1.25), prob=0.5):
    factor = tf.random.uniform([], minval=saturation_factor[0], maxval=saturation_factor[1], dtype=tf.float32)
    return apply_func_with_prob(F.adjust_saturation, image, (factor, ), prob), mask


def random_gamma(image, mask, gamma_range=(0.75, 1.25), gain=1.0, prob=0.5):
    gamma = tf.random.uniform([], minval=gamma_range[0], maxval=gamma_range[1], dtype=tf.float32)
    return apply_func_with_prob(F.adjust_gamma, image, (gamma, gain), prob), mask


def random_jpeg_quality(image, mask, jpeg_quality_range=(75, 95), prob=0.5):
    jpeg_quality = tf.random.uniform([], minval=jpeg_quality_range[0], maxval=jpeg_quality_range[1], dtype=tf.int32)
    return apply_func_with_prob(F.adjust_jpeg_quality, image, (jpeg_quality, ), prob), mask


def random_elastic_deform(image, mask, scale=10, strength=10, mask_max=255, prob=0.5):
    def elastic_deform(image, mask, scale, strength, mask_max):
        size = tf.cast(tf.shape(image), tf.int32)
        flow = tf.random.uniform([tf.math.floordiv(size[0], scale),
                                tf.math.floordiv(size[1], scale),
                                2], -1, 1)
        flow = tfa.image.gaussian_filter2d(flow, filter_shape=(3, 3), sigma=5) * strength
        flow = tf.image.resize(flow, size[0:2])
        
        image = tfa.image.dense_image_warp(tf.expand_dims(tf.cast(image, tf.float32), axis=0), tf.expand_dims(flow, axis=0))[0]
        mask = tfa.image.dense_image_warp(tf.expand_dims(tf.cast(mask, tf.float32), axis=0), tf.expand_dims(flow, axis=0))[0]
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, tf.uint8)
        mask = tf.where(mask>mask_max//2,mask_max,0)
        mask = tf.cast(mask, tf.uint8)
        return image, mask
    return tf.cond(
        tf.random.uniform([], 0, 1) < prob,
        lambda: elastic_deform(image, mask, scale=scale, strength=strength, mask_max=mask_max),
        lambda: (image, mask),
    )
    
def random_sparse_warp(image, mask, dst_x=0.3, dst_y=0.3, mask_max=255, prob=0.5):
    def _random_sparse_warp(image, mask, dst_x, dst_y, mask_max):
        size = tf.cast(tf.shape(image), tf.float32)
        src_points = tf.cast(tf.convert_to_tensor([[[0,0],[0,size[0]],[size[1],0],[size[1], size[0]]]]), tf.float32)
        shift_x = tf.random.uniform([1,4], -size[1]*dst_x, size[1]*dst_x) + tf.cast(tf.convert_to_tensor([[0,0,size[1],size[1]]]), tf.float32)
        shift_y = tf.random.uniform([1,4], -size[0]*dst_y, size[0]*dst_y) + tf.cast(tf.convert_to_tensor([[0,size[0],0,size[0]]]), tf.float32)

        dst_points = tf.stack([shift_x, shift_y], axis=2)
        image, flow = tfa.image.sparse_image_warp(image=tf.cast(tf.convert_to_tensor(image), tf.float32), 
                                            source_control_point_locations=src_points,
                                            dest_control_point_locations=dst_points)
        mask = tfa.image.dense_image_warp(tf.expand_dims(tf.cast(mask, tf.float32), axis=0), flow)[0]
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, tf.uint8)
        mask = tf.where(mask>mask_max//2,mask_max,0)
        mask = tf.cast(mask, tf.uint8)
        return image, mask
    return tf.cond(
        tf.random.uniform([], 0, 1) < prob,
        lambda: _random_sparse_warp(image, mask, dst_x=dst_x, dst_y=dst_y, mask_max=mask_max),
        lambda: (image, mask),
    )

def random_gaussian_noise(image, mask, stddev_range=[5, 95], prob=0.5):
    def gaussian_noise(image, stddev_range):
        image = tf.cast(image, tf.float32)
        stddev = tf.random.uniform([], minval=stddev_range[0], maxval=stddev_range[1], dtype=tf.float32)
        image = tf.random.normal(tf.shape(image), stddev=stddev) + image
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, tf.uint8)
        return image
    return tf.cond(
        tf.random.uniform([], 0, 1) < prob,
        lambda: (gaussian_noise(image, stddev_range=stddev_range), mask),
        lambda: (image, mask),
    )

def random_speckle_noise(image, mask, prob_range=[0.0, 0.05], prob=0.5):
    def speckle_noise(image, prob_range):
        image = tf.cast(image, tf.float32)
        prob = tf.random.uniform([], minval=prob_range[0], maxval=prob_range[1], dtype=tf.float32)
        sample = tf.random.uniform(tf.shape(image))
        image = tf.where(sample <= prob, tf.cast(tf.zeros_like(image), tf.float32), image)
        image = tf.where(sample >= (1. - prob), 255.*tf.cast(tf.ones_like(image), tf.float32), image)
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, tf.uint8)
        return image
    return tf.cond(
        tf.random.uniform([], 0, 1) < prob,
        lambda: (speckle_noise(image, prob_range=prob_range), mask),
        lambda: (image, mask),
    )

def random_crop(image, mask, size=(256,256), prob=0.5):
    def _random_crop(image, mask, size):
        shape = tf.shape(image)
        size = tf.convert_to_tensor([size[0], size[1], shape[2]])
        limit = shape - size + 1
        offset = tf.random.uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max) % limit
        image = tf.slice(image, offset, size)
        mask = tf.slice(mask, offset, size)
        return image, mask
    return tf.cond(
        tf.random.uniform([], 0, 1) < prob,
        lambda: _random_crop(image, mask, size),
        lambda: (tf.cast(tf.image.resize(image, size=size), tf.uint8), tf.cast(tf.image.resize(mask, size=size), tf.uint8)),
    )