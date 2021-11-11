""" Loader Factory, Fast Collate, CUDA Prefetcher

Prefetcher and Fast Collate inspired by NVIDIA APEX example at
https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf

Hacked together by / Copyright 2021 Ross Wightman
"""

import os

import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset import ImageFolderDataset, MindDataset

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .mixup import Mixup
from .transforms_factory import create_transform


def imagenet_images(dataset_dir, num_parallel_workers, shuffle):
    num_shards, rank_id = _get_rank_info()
    dataset = ImageFolderDataset(dataset_dir, num_parallel_workers=num_parallel_workers,
                                 shuffle=shuffle, num_shards=num_shards, shard_id=rank_id)
    return dataset


def imagenet_mind(dataset_dir, num_parallel_workers, shuffle):
    num_shards, rank_id = _get_rank_info()
    files = os.listdir(dataset_dir)
    data_file = list(filter(lambda x: not x.endswith(".db"), files))
    if len(data_file) == 1:
        data_file = data_file[0]
    else:
        data_file = list(filter(lambda x: x.endswith("0"), data_file))[0]
    data_file = os.path.join(dataset_dir, data_file)
    dataset = MindDataset(data_file, num_parallel_workers=num_parallel_workers,
                          shuffle=shuffle, columns_list=["image", "label"],
                          num_shards=num_shards, shard_id=rank_id)
    return dataset


_imagenet_parser = {
    "image": imagenet_images,
    "mindrecord": imagenet_mind
}


def create_imagenet_loader(
        root,
        batch_size,
        input_size,
        num_classes,
        is_training=False,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_splits=0,
        interpolation='bilinear',
        cutmix=0.,
        mix_up=0.,
        mixup_prob=0.,
        switch_prob=0.,
        mixup_mode="batch",
        label_smoothing=0.1,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        crop_pct=None,
        num_parallel_workers=8,
        data_type="image",
):
    assert data_type in ["image", "mindrecord"]
    dataset = _imagenet_parser[data_type](
        root=root, num_parallel_workers=num_parallel_workers, shuffle=is_training)
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    transform_image = create_transform(
        input_size,
        is_training=is_training,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )
    transform_label = C.TypeCast(mstype.int32)

    dataset = dataset.map(input_columns="image", num_parallel_workers=num_parallel_workers,
                          operations=transform_image)
    dataset = dataset.map(input_columns="label", num_parallel_workers=num_parallel_workers,
                          operations=transform_label)
    if (mix_up > 0. or cutmix > 0.) and not is_training:
        # if use mixup and not training(False), one hot val data label
        one_hot = C.OneHot(num_classes=num_classes)
        dataset = dataset.map(input_columns="label", num_parallel_workers=num_parallel_workers,
                              operations=one_hot)
    dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_workers=num_parallel_workers)
    if (mix_up > 0. or cutmix > 0.) and is_training:
        mixup_fn = Mixup(
            mixup_alpha=mix_up, cutmix_alpha=cutmix, cutmix_minmax=None,
            prob=mixup_prob, switch_prob=switch_prob, mode=mixup_mode,
            label_smoothing=label_smoothing, num_classes=num_classes)

        dataset = dataset.map(operations=mixup_fn, input_columns=["image", "label"],
                              num_parallel_workers=num_parallel_workers)
    return dataset


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
