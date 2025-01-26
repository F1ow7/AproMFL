"""libaray for multi-modal dataset loaders.

Acknowledgements:
`image_to_caption_collate_fn` is based on
https://github.com/yalesong/pvse/blob/master/data.py
"""
import os

import numpy as np

import torch
from torch.utils.data import DataLoader
from src.datasets.transforms import imagenet_transform, caption_transform
from src.datasets.vocab import Vocabulary
from src.datasets.flickr30k import F30kCaptionsCap,Flickr
# from . import _utils
from torch.utils.data import default_collate
# default_collate: _collate_fn_t = _utils.collate.default_collate

def image_captions_collate_fn(batch):
    transposed = list(zip(*batch))
    imgs = default_collate(transposed[0])
    texts = transposed[1]
    return imgs, texts

def image_to_caption_collate_fn(data):
    """Build mini-batch tensors from a list of (image, sentence) tuples.
    Args:
      data: list of (image, sentence) tuple.
        - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
        - sentence: torch tensor of shape (?); variable length.

    Returns:
      images: torch tensor of shape (batch_size, 3, 256, 256) or
              (batch_size, padded_length, 3, 256, 256).
      targets: torch tensor of shape (batch_size, padded_length).
      lengths: list; valid length for each padded sentence.
    """
    # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, sentences, captions, ann_ids, image_ids, index = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = [len(cap) for cap in sentences]
    targets = torch.zeros(len(sentences), max(cap_lengths)).long()
    for i, cap in enumerate(sentences):
        end = cap_lengths[i]
        targets[i, :end] = cap[:end]

    cap_lengths = torch.Tensor(cap_lengths).long()
    # print('cap_lengths', type(cap_lengths))
    return images, targets, captions, cap_lengths, ann_ids, image_ids, index


def load_vocab(vocab_path):
    if isinstance(vocab_path, str):
        vocab = Vocabulary()
        vocab.load_from_pickle(vocab_path)
    else:
        vocab = vocab_path
    return vocab

def _get_F30k_loader(
                     num_workers,
                     batch_size=64,
                     train=False,
                     split='train',
                     cutout_prob=0.0,
                     caption_drop_prob=0.0,
                     client=-1):
    _image_transform = imagenet_transform(
        random_resize_crop=train,
        random_erasing_prob=cutout_prob,
    )
    _caption_transform = None
    root='/autodl-fs/data/yClient/mmdata/Flickr30k'
    annotation_file = f"{root}/flickr30k_{split}_karpathy.txt"
    flickr30k_dataset = Flickr(root=root, ann_file=annotation_file, transform=_image_transform, target_transform =_caption_transform, client=client, train=train)

    # flickr30k_dataset = F30kCaptionsCap(train=True if split == 'train' else False,
    #                                transform=_image_transform,
    #                                target_transform=_caption_transform, client=client)
    print(f'f30k train {len(flickr30k_dataset)}')
    
    dataloader = DataLoader(flickr30k_dataset,
                            batch_size=batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            collate_fn=image_captions_collate_fn,
                            pin_memory=True)
    print(f'Loading F30k Caption: n_images {len(flickr30k_dataset)}...')
    return dataloader


def prepare_f30k_dataloaders(dataloader_config,
                             client=-1,
                             num_workers=6):
    """Prepare MS-COCO Caption train / val / test dataloaders
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/coco_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "te"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """
    batch_size = dataloader_config['batch_size']
    tr_cutout_prob = dataloader_config.get('random_erasing_prob', 0.0)
    tr_caption_drop_prob = dataloader_config.get('caption_drop_prob', 0.0)
    eval_batch_size = dataloader_config.get('eval_batch_size', batch_size)

    # vocab = load_vocab(vocab_path)

    dataloaders = {}
    dataloaders['train'] = _get_F30k_loader(
        num_workers=num_workers,
        batch_size=batch_size,
        train=True,
        split='train',
        cutout_prob=tr_cutout_prob,
        caption_drop_prob=tr_caption_drop_prob,
        client=client
    )

    # dataloaders['val'] = _get_F30k_loader(
    #     vocab,
    #     num_workers=num_workers,
    #     batch_size=eval_batch_size,
    #     train=False,
    #     split='val',
    #     client=client
    #
    # )

    dataloaders['test'] = _get_F30k_loader(
        num_workers=num_workers,
        batch_size=eval_batch_size,
        train=False,
        split='test',
        client=client
    )
    

    return dataloaders


def see_f30k_len():
    train = F30kCaptionsCap(split='train')

    test = F30kCaptionsCap(split='test')

    print(f'f30k train {len(train)}')
    print(f'f30k test {len(test)}')