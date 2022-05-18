import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import math

@tf.function
def cutout(image, pad_size, cutout_center_height, cutout_center_width, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.
    Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that
      is applied to the image. The mask will be of size
      (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has
      the cutout mask applied to it.
    Returns:
    An image Tensor that is of type uint8.
    """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_channel = tf.shape(image)[2]
    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [image_height - (lower_pad + upper_pad), image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, image_channel])
    image = tf.where(tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype) * replace, image)
    return image


@tf.function
def solarize(image, threshold=128):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return tf.where(image < threshold, image, 255 - image)


@tf.function
def solarize_add(image, addition=0, threshold=128):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = tf.cast(image, tf.int64) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
    return tf.where(image < threshold, added_image, image)


@tf.function
def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.
    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.
    Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.
    Returns:
    A blended image Tensor of type uint8.
    """
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, tf.float32) + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


@tf.function
def color(image, factor):
    """Equivalent of PIL Color."""
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return tf.cast(blend(degenerate, image, factor), tf.uint8)


@tf.function
def contrast(image, factor):
    """Equivalent of PIL Contrast."""
    degenerate = tf.image.rgb_to_grayscale(image)
    # Cast before calling tf.histogram.
    degenerate = tf.cast(degenerate, tf.int32)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
    mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
    degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
    return tf.cast(tfa.image.blend(degenerate, image, factor), tf.uint8)


@tf.function
def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image)
    return tf.cast(tfa.image.blend(degenerate, image, factor), tf.uint8)


@tf.function
def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    shift = 8 - bits
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


@tf.function
def rotate(image, degrees, interpolation="nearest", fill_mode="constant", fill_value=0.0):
    """Rotates the image by degrees either clockwise or counterclockwise.
    Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.
    replace: A one or three value 1D tensor to fill empty pixels caused by
      the rotate operation.
    Returns:
    The rotated version of image.
    """
    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians

    # In practice, we should randomize the rotation degrees by flipping
    # it negatively half the time, but that's done on 'degrees' outside
    # of the function.
    return tfa.image.rotate(image, radians, interpolation=interpolation, fill_mode=fill_mode, fill_value=fill_value)


@tf.function
def invert(image):
    """Inverts the image pixels."""
    image = tf.convert_to_tensor(image)
    return 255 - image


@tf.function
def equalize(image):
    """Implements Equalize function from PIL using TF ops."""

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0), lambda: im, lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image


@tf.function
def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
    orig_image = image
    image = tf.cast(image, tf.float32)
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)
    # SMOOTH PIL Kernel.
    kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]) / 13.0
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(image, kernel, strides, padding="VALID", dilations=[1, 1])
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    return tf.cast(blend(result, orig_image, factor), tf.uint8)


@tf.function
def autocontrast(image):
    """Implements Autocontrast function from PIL using TF ops.
    Args:
    image: A 3D uint8 tensor.
    Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
    """

    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: tf.cast(image, tf.uint8))
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return image


@tf.function
def translate_x(image, pixels, interpolation='nearest', fill_mode='constant', fill_value=0.0):
    """Equivalent of PIL Translate in X dimension."""
    return tfa.image.translate(image, [pixels, 0], interpolation=interpolation, fill_mode=fill_mode, fill_value=fill_value)


@tf.function
def translate_y(image, pixels, interpolation='nearest', fill_mode='constant', fill_value=0.0):
    """Equivalent of PIL Translate in Y dimension."""
    return tfa.image.translate(image, [0, pixels], interpolation=interpolation, fill_mode=fill_mode, fill_value=fill_value)


@tf.function
def shear_x(image, level, interpolation='nearest', fill_mode='constant', fill_value=0.0):
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    #return tfa.image.shear_x(image, level, replace)
    return tfa.image.transform(image, [1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    , interpolation=interpolation, fill_mode=fill_mode, fill_value=fill_value)


@tf.function
def shear_y(image, level, interpolation='nearest', fill_mode='constant', fill_value=0.0):
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    #return tfa.image.shear_y(image, level, replace)
    return tfa.image.transform(image, [1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0]
    , interpolation=interpolation, fill_mode=fill_mode, fill_value=fill_value)


@tf.function
def scale_xy(image, scale, interop, fill_mode, fill_value):
    """Zoom In/Out"""
    image = tf.convert_to_tensor(image)
    h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
    translate_x = 0.5 * (w - (w * scale[0]))
    translate_y = 0.5 * (h - (h * scale[1]))
    matrix = [scale[0], 0.0, translate_x, 0.0, scale[1], translate_y, 0.0, 0.0]
    return tfa.image.transform(image, matrix, interpolation=interop, fill_mode=fill_mode, fill_value=fill_value)

@tf.function
def grid_shuffle(image, grid_x, grid_y, grid_size, order):
    """Grid Shuffle"""
    image = tf.convert_to_tensor(image)
    h = tf.cast(tf.shape(image)[0], tf.int32)
    w = tf.cast(tf.shape(image)[1], tf.int32)
    c = tf.cast(tf.shape(image)[2], tf.int32)
    pad_h = h - grid_y * grid_size[1]
    pad_w = w - grid_x * grid_size[0]
    mask = tf.ones([grid_y * grid_size[1], grid_x * grid_size[0], c], dtype=tf.bool)
    mask = tf.pad(mask, [[0, pad_h], [0, pad_w], [0, 0]])
    block = [ None ] * grid_size[1]
    index = 0
    for j in range(grid_size[1]):
        rows = [ None ] * grid_size[0]
        for i in range(grid_size[0]):
            k = order[index]
            row, col = k // grid_size[0], k % grid_size[0]
            rows[i] = tf.slice(image, [grid_y * row, grid_x * col, 0], [grid_y, grid_x, c])
            index = index + 1
        block[j] = tf.concat(rows, axis=1)
    out = tf.concat(block, axis=0)
    out = tf.pad(out, [[0, pad_h], [0, pad_w], [0, 0]])
    out = tf.where(mask == True, out, image)
    return out


@tf.function
def affine(image, trans_x=0.0, trans_y=0.0, shear_x=0.0, shear_y=0.0, scale_x=1.0, scale_y=1.0, degree=0.0, interpolation='nearest', fill_mode='constant', fill_value=0):
    size = tf.cast(tf.shape(image), tf.float32)
    inv_cent = [[1.0, 0.0, -(size[1]/2+0.5)],
                [0.0, 1.0, -(size[0]/2+0.5)],
                [0.0, 0.0, 1.0]]
    
    cent = [[1.0, 0.0, size[1]/2+0.5],
            [0.0, 1.0, size[0]/2+0.5],
            [0.0, 0.0, 1.0]]
    
    translate_matrix = [[1.0, 0.0, trans_x],
                        [0.0, 1.0, trans_y],
                        [0.0, 0.0, 1.0]]

    shear_matrix = [[1.0, shear_x, 0.0],
                    [shear_y, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]

    scale_matrix = [[scale_x, 0.0, 0.0],
                    [0.0, scale_y, 0.0],
                    [0.0, 0.0, 1.0]]

    rad = tf.constant(0.01745329251) * degree
    rotate_matrix = [[tf.math.cos(rad), tf.math.sin(rad), 0.0],
                     [-tf.math.sin(rad), tf.math.cos(rad), 0.0],
                     [0.0, 0.0, 1.0]]
    
    
    matrix = tf.matmul(translate_matrix, inv_cent)
    matrix = tf.matmul(shear_matrix, matrix)
    matrix = tf.matmul(scale_matrix, matrix)
    matrix = tf.matmul(rotate_matrix, matrix)
    matrix = tf.matmul(cent, matrix)
    
    transforms = tf.reshape(matrix, tf.constant([-1, 9]))
    transforms /= transforms[:, 8:9]
    transforms = transforms[:, :8]
    return tfa.image.transform(image, transforms, interpolation=interpolation, fill_mode=fill_mode, fill_value=fill_value)

@tf.function
def adjust_hue(image, delta):
    return tf.image.adjust_hue(image, delta)

@tf.function
def adjust_saturation(image, saturation_factor):
    return tf.image.adjust_saturation(image, saturation_factor)

@tf.function
def adjust_gamma(image, gamma, gain):
    return tf.image.adjust_gamma(image, gamma, gain)

@tf.function
def adjust_jpeg_quality(image, jpeg_quality):
    return tf.image.adjust_jpeg_quality(image, jpeg_quality)
