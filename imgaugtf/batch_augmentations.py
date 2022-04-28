import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import math

def mixup(image, label, alpha=0.5):
    batch_size = tf.shape(image)[0]
    image = tf.cast(image, tf.float32)
    mix_weight = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    mix_weight = tf.maximum(mix_weight, 1.0 - mix_weight)
    img_weight = tf.cast(tf.reshape(mix_weight, [batch_size, 1, 1, 1]), image.dtype)
    # Mixup on a single batch is implemented by taking a weighted sum with the
    # same batch in reverse.
    image = image * img_weight + image[::-1] * (1.0 - img_weight)
    label_weight = tf.cast(mix_weight, label.dtype)
    label = label * label_weight + label[::-1] * (1 - label_weight)

    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, dtype=tf.uint8)
    return image, label


def cutmix_mask(alpha, h, w):
    """Returns image mask for CutMix."""
    r_x = tf.random.uniform([], 0, w, tf.int32)
    r_y = tf.random.uniform([], 0, h, tf.int32)

    area = tfp.distributions.Beta(alpha, alpha).sample()
    patch_ratio = tf.cast(tf.math.sqrt(1 - area), tf.float32)
    r_w = tf.cast(patch_ratio * tf.cast(w, tf.float32), tf.int32)
    r_h = tf.cast(patch_ratio * tf.cast(h, tf.float32), tf.int32)
    bbx1 = tf.clip_by_value(tf.cast(r_x - r_w // 2, tf.int32), 0, w)
    bby1 = tf.clip_by_value(tf.cast(r_y - r_h // 2, tf.int32), 0, h)
    bbx2 = tf.clip_by_value(tf.cast(r_x + r_w // 2, tf.int32), 0, w)
    bby2 = tf.clip_by_value(tf.cast(r_y + r_h // 2, tf.int32), 0, h)

    # Create the binary mask.
    pad_left = bbx1
    pad_top = bby1
    pad_right = tf.maximum(w - bbx2, 0)
    pad_bottom = tf.maximum(h - bby2, 0)
    r_h = bby2 - bby1
    r_w = bbx2 - bbx1

    mask = tf.pad(tf.ones((r_h, r_w)), paddings=[[pad_top, pad_bottom], [pad_left, pad_right]], mode="CONSTANT", constant_values=0)
    # mask.set_shape((h, w))
    return mask[..., None]  # Add channel dim.


def cutmix(image, label):
    """Applies CutMix regularization to a batch of images and labels.
    Reference: https://arxiv.org/pdf/1905.04899.pdf
    Arguments:
        image: a Tensor of batched images.
        label: a Tensor of batched one-hot labels.
    Returns:
        Updated images and labels with the same dimensions as the input
        with CutMix regularization applied.
    """
    image = tf.cast(image, tf.float32)
    image_height, image_width = tf.shape(image)[1], tf.shape(image)[2]
    mask = cutmix_mask(alpha=1.0, h=image_height, w=image_width)

    # actual area of cut & mix pixels
    mix_area = tf.reduce_sum(mask) / tf.cast(tf.size(mask), mask.dtype)
    mask = tf.cast(mask, image.dtype)
    mixed_image = (1.0 - mask) * image + mask * image[::-1]
    mix_area = tf.cast(mix_area, label.dtype)
    mixed_label = (1.0 - mix_area) * label + mix_area * label[::-1]

    mixed_image = tf.clip_by_value(mixed_image, 0, 255)
    mixed_image = tf.cast(mixed_image, dtype=tf.uint8)
    return mixed_image, mixed_label
