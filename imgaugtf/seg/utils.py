import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import math

from . import augmentations as aug

PIXEL_OPERATORS = [
    {"func": aug.random_cutout, "option": {"num_holes": 8, "hole_size": 20, "replace": 0}},
    {"func": aug.random_solarize, "option": {"threshold": 128}},
    {"func": aug.random_solarize_add, "option": {"addition": 30, "threshold": 128}},
    {"func": aug.random_color, "option": {"alpha_range": (0.2, 0.8)}},
    {"func": aug.random_contrast, "option": {"lower": 0.2, "upper": 0.8}},
    {"func": aug.random_brightness, "option": {"max_delta": 0.1}},
    {"func": aug.random_posterize, "option": {"bits": 4}},
    {"func": aug.random_invert, "option": {}},
    {"func": aug.random_equalize, "option": {"bins": 256}},
    {"func": aug.random_sharpness, "option": {"alpha_range": (-3.0, 3.0)}},
    {"func": aug.random_autocontrast, "option": {}},
    {"func": aug.random_hsv_in_yiq, "option": {"max_delta_hue": 0.2, "lower_saturation": 0.5, "upper_saturation": 1.0, "lower_value": 0.5, "upper_value": 1.0}},
    {"func": aug.random_gaussian_filter2d, "option": {"filter_shape": (3, 3)}},
    {"func": aug.random_mean_filter2d, "option": {"filter_shape": (3, 3)}},
    {"func": aug.random_median_filter2d, "option": {"filter_shape": (3, 3)}},
    {"func": aug.random_gray, "option": {}},
    {"func": aug.random_hue, "option": {"max_delta": 0.2}},
    {"func": aug.random_saturation, "option": {"saturation_factor": (0.75, 1.25)}},
    {"func": aug.random_gamma, "option": {"gamma_range": (0.75, 1.25)}},
    {"func": aug.random_jpeg_quality, "option": {"jpeg_quality_range": (75, 95)}},
    {"func": aug.random_gaussian_noise, "option": {"stddev_range": (5, 95)}},
    {"func": aug.random_speckle_noise, "option": {"prob_range": (0.0, 0.05)}},
]
GEO_OPERATORS = [
    {"func": aug.random_flip_left_right, "option": {}},
    {"func": aug.random_flip_up_down, "option": {}},
    {"func": aug.random_rotate, "option": {"degree_range": (-90, 90), "interpolation": "nearest", "fill_mode": "constant", "fill_value": 0.0}},
    {"func": aug.random_translate_x, "option": {"percent": 0.5, 'interpolation': 'nearest', 'fill_mode': 'constant', 'fill_value': 0.0}},
    {"func": aug.random_translate_y, "option": {"percent": 0.5, 'interpolation': 'nearest', 'fill_mode': 'constant', 'fill_value': 0.0}},
    {"func": aug.random_shear_x, "option": {"percent": 0.3, 'interpolation': 'nearest', 'fill_mode': 'constant', 'fill_value': 0.0}},
    {"func": aug.random_shear_y, "option": {"percent": 0.3, 'interpolation': 'nearest', 'fill_mode': 'constant', 'fill_value': 0.0}},
    {"func": aug.random_zoom, "option": {"scale": (-0.2, 0.2), "interpolation": "nearest", "fill_mode": "constant", "fill_value": 0}},
    {"func": aug.random_grid_shuffle, "option": {"grid_size": (3, 3)}},
    {"func": aug.random_affine, "option": {"translate": (-0.3, 0.3), "shear": (-0.3, 0.3), "rotate": (-90, 90), "scale": (0.75, 1.25)
    , "interpolation": 'nearest', "fill_mode": 'constant', "fill_value": 0}},
    {"func": aug.random_elastic_deform, "option": {"scale": 10, "strength": 10}},
    {"func": aug.random_sparse_warp, "option": {"dst_x": 0.3, "dst_y": 0.3}},
]

OPERATORS = PIXEL_OPERATORS + GEO_OPERATORS


def apply_one(image, mask, functions=OPERATORS, prob=1.0):
    def _apply_one(image, mask, functions):
        op_to_select = tf.random.uniform([], maxval=len(functions), dtype=tf.int32)
        for (i, op) in enumerate(functions):
            image, mask = tf.cond(tf.equal(i, op_to_select), lambda: op["func"](image, mask, prob=1.0, **op["option"]), lambda: (image, mask))
        return image, mask

    return tf.cond(tf.random.uniform([], 0, 1) < prob, lambda: _apply_one(image, mask, functions=functions), lambda: (image, mask))


def apply_n(image, mask, functions=OPERATORS, num_ops=2, prob=1.0):
    def _apply_n(image, mask, functions, num_ops, prob):
        for i in range(num_ops):
            image, mask = apply_one(image, mask, functions=functions, prob=prob)
        return image, mask

    return _apply_n(image, mask, functions=functions, num_ops=num_ops, prob=prob)
