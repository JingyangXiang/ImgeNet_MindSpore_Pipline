""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2020 Ross Wightman
"""
import math

import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.vision.py_transforms as py_transform
from mindspore.dataset.transforms.py_transforms import Compose

from .auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from .random_erasing import RandomErasing
from .transforms import _pil_interp, RandomResizedCropAndInterpolation


def transforms_noaug_train(
        img_size=224,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
):
    if interpolation == 'random':
        # random interpolation not supported with no-aug
        interpolation = 'bilinear'
    tfl = [
        vision.Decode(),
        py_transform.ToPIL(),
        py_transform.Resize(img_size, _pil_interp(interpolation)),
        py_transform.CenterCrop(img_size)
    ]
    tfl += [
        py_transform.ToTensor(),
        py_transform.Normalize(
            mean=mean,
            std=std)
    ]
    return Compose(tfl)


def transforms_imagenet_train(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3. / 4., 4. / 3.))  # default imagenet ratio range
    primary_tfl = [
        vision.Decode(),
        py_transform.ToPIL(),
        RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.:
        primary_tfl += [py_transform.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [py_transform.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [py_transform.ColorJitter(*color_jitter)]

    final_tfl = []

    final_tfl += [
        py_transform.ToTensor(),
        py_transform.Normalize(
            mean=mean,
            std=std)
    ]
    if re_prob > 0.:
        final_tfl.append(
            RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits))

    if separate:
        return py_transform.Compose(primary_tfl), py_transform.Compose(secondary_tfl), py_transform.Compose(final_tfl)
    else:
        return py_transform.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_imagenet_eval(
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        vision.Decode(),
        py_transform.ToPIL(),
        py_transform.Resize(scale_size, _pil_interp(interpolation)),
        py_transform.CenterCrop(img_size),
    ]

    tfl += [
        py_transform.ToTensor(),
        py_transform.Normalize(
            mean=mean,
            std=std)
    ]

    return py_transform.Compose(tfl)


def create_transform(
        input_size,
        is_training=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        separate=False):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if is_training and no_aug:
        assert not separate, "Cannot perform split augmentation with no_aug"
        transform = transforms_noaug_train(
            img_size,
            interpolation=interpolation,
            mean=mean,
            std=std)
    elif is_training:
        transform = transforms_imagenet_train(
            img_size,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            mean=mean,
            std=std,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            separate=separate)
    else:
        assert not separate, "Separate transforms not supported for validation preprocessing"
        transform = transforms_imagenet_eval(
            img_size,
            interpolation=interpolation,
            mean=mean,
            std=std,
            crop_pct=crop_pct)

    return transform
