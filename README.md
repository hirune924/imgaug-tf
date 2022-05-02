# imgaug-tf

An image augmentation library for tensorflow. 
This library is implemented in TF native and has been confirmed to work with TPU.

## Installation
```bash
%env TOKEN=[your secret access token]
!pip install -U git+https://$$TOKEN@github.com/hirune924/imgaug-tf
```
Required packages:
- tensorflow (2.6.3 or higher recommended)
- tensorflow_addons (0.14.0 or higher recommended)
- tensorflow_probability (0.14.1 or higher recommended)

## Quick Start
imgaugtf is implemented to work simply with tf.data.
Example of use single transform.
```python
import imgaugtf

image = imgaugtf.random_solarize_add(image, addition=30, prob=0.5)
```
You can also apply transform for a mask as same as a image.
```python
import imgaugtf

image = imgaugtf.seg.random_solarize_add(image, mask, addition=30, prob=0.5)
```
You can also randomly select n of multiple transformations to apply, as shown below. You can use mixup or cutmix on batched images.
```python
import imgaugtf

def augmentation(example):
    example['image'] = imgaugtf.random_resized_crop(example['image'], size=[256, 256], prob=1.0)
    example['image'] = imgaugtf.apply_n(example['image'], functions=imgaugtf.OPERATORS, num_ops=2, prob=1.0)
    return example

def batch_augmentation(example, num_class=120):
    image = example['image']
    label = tf.one_hot(example['label'], num_class)
    
    image, label = imgaugtf.cutmix(image, label)
    return image, label

result = next(iter(dataset.map(augmentation).batch(15).map(batch_augmentation)))

for i in range(10):
    plt.imshow(result[0][i])
    plt.show()
```
functions is list of dict like this example. dict has keys of 'func' and 'option'. you can customize it you like.
```python
[
    {"func": imgaugtf.random_cutout, "option": {"num_holes": 8, "hole_size": 20, "replace": 0}},
    {"func": imgaugtf.random_solarize, "option": {"threshold": 128}},
    {"func": imgaugtf.random_solarize_add, "option": {"addition": 30, "threshold": 128}},
]
```

## Augmentations
### pixel
|  | image | mask |
| :---: | :---: | :---: |
| original | ![original](./images/deer_org.png) | ![original](./images/deer_mask_org.png) |
| random_solarize | ![random_solarize](./images/random_solarize.png) | ![original](./images/deer_mask_org.png) |
| random_solarize_add | ![random_solarize_add](./images/random_solarize_add.png) | ![original](./images/deer_mask_org.png) |
| random_color |![random_color](./images/random_color.png) | ![original](./images/deer_mask_org.png) |
|  random_contrast |![random_contrast](./images/random_contrast.png) | ![original](./images/deer_mask_org.png) |
| random_brightness | ![random_brightness](./images/random_brightness.png)| ![original](./images/deer_mask_org.png) |
| random_posterize |![random_posterize](./images/random_posterize.png) | ![original](./images/deer_mask_org.png) |
| random_invert |![random_invert](./images/random_invert.png) |![original](./images/deer_mask_org.png)  |
| random_equalize | ![random_equalize](./images/random_equalize.png) | ![original](./images/deer_mask_org.png) |
| random_sharpness |![random_sharpness](./images/random_sharpness.png) | ![original](./images/deer_mask_org.png) |
| random_autocontrast | ![random_autocontrast](./images/random_autocontrast.png) | ![original](./images/deer_mask_org.png) |
| random_hsv_in_yiq |![random_hsv_in_yiq](./images/random_hsv_in_yiq.png) | ![original](./images/deer_mask_org.png) |
| random_gaussian_filter2d |![random_gaussian_filter2d](./images/random_gaussian_filter2d.png) | ![original](./images/deer_mask_org.png) |
| random_mean_filter2d | ![random_mean_filter2d](./images/random_mean_filter2d.png)| ![original](./images/deer_mask_org.png) |
| random_median_filter2d |![random_median_filter2d](./images/random_median_filter2d.png) | ![original](./images/deer_mask_org.png) |
| random_cutout | ![random_cutout](./images/random_cutout.png) | ![original](./images/deer_mask_org.png) |
| random_gray | ![random_gray](./images/random_gray.png) | ![original](./images/deer_mask_org.png) |




### geometory
|  | image | mask |
| :---: | :---: | :---: |
| original | ![original](./images/deer_org.png) | ![original](./images/deer_mask_org.png) |
| random_flip_left_right | ![random_flip_left_right](./images/random_flip_left_right.png) | ![original](./images/random_flip_left_right_mask.png) |
| random_flip_up_down | ![random_flip_up_down](./images/random_flip_up_down.png) | ![original](./images/random_flip_up_down_mask.png) |
| random_resized_crop | ![random_resized_crop](./images/random_resized_crop.png) | ![original](./images/random_resized_crop_mask.png) |
| random_rotate | ![random_rotate](./images/random_rotate.png) | ![original](./images/random_rotate_mask.png) |
| random_translate_x | ![random_translate_x](./images/random_translate_x.png) | ![original](./images/random_translate_x_mask.png) |
| random_translate_y | ![random_translate_y](./images/random_translate_y.png) | ![original](./images/random_translate_y_mask.png) |
| random_shear_x | ![random_shear_x](./images/random_shear_x.png) | ![original](./images/random_shear_x_mask.png) |
| random_shear_y | ![random_shear_y](./images/random_shear_y.png) | ![original](./images/random_shear_y_mask.png) |
| random_zoom | ![random_zoom](./images/random_zoom.png) | ![original](./images/random_zoom_mask.png) |
| random_grid_shuffle | ![random_grid_shuffle](./images/random_grid_shuffle.png) | ![original](./images/random_grid_shuffle_mask.png) |


### blend
* mixup
* cutmix

### compose
* apply_one
* apply_n

