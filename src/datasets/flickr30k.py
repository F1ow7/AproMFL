import os

from src.datasets.transforms import caption_transform
from src.datasets.vocab import Vocabulary

try:
    import ujson as json
except ImportError:
    import json

import numpy as np

from PIL import Image
from pycocotools.coco import COCO

from torch.utils.data import Dataset
from glob import glob
import pickle
# from dataset.vocab import Vocabulary
# from dataset.coco_transforms import imagenet_transform, caption_transform
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.datasets import VisionDataset

class Flickr(VisionDataset):

    def __init__(
        self,
        root: str,
        ann_file: str,
        
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_iid=False, client=-1,train=True
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        data = defaultdict(list)
        with open(ann_file) as fd:
            fd.readline()
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    img, caption = line.strip().split(".jpg,")
                    img = img + ".jpg"
                    data[img].append(caption)
        self.data = list(data.items())
        if client > -1 and train:
            # print(self.data)
            indices = self.iid()[client] if is_iid else self.non_iid()[client]
            indices = np.array(list(indices)).astype(int)

            self.data = [self.data[i] for i in indices]
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img, captions = self.data[index]

        # Image
        img = Image.open(os.path.join(self.root+"/flickr30k-images/", img)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target =  captions
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)
    def iid(self, root='/autodl-fs/data/yClient/mmdata/Flickr30k/', num_users=20):
        """
        Sample I.I.D. client data from MNIST dataset
        :param dataset:
        :param num_users:
        :return: dict of image index
        """
        pkl_path = root + 'client_iid.pkl'
        if os.path.exists(pkl_path):
            dict_users = pickle.load(open(pkl_path, 'rb'))
        else:
            num_items = int(len(self.data) / num_users)
            dict_users, all_idxs = {}, [i for i in range(len(self.data))]
            for i in range(num_users):
                dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                     replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])
            pickle.dump(dict_users, open(pkl_path, 'wb'))
        return dict_users

    def non_iid(self, root='/root/newMFL/data_partition/', num_users=2):
        pkl_path = root + f'{num_users}client_noniid_flickr30k.pkl'
        if os.path.exists(pkl_path):
            dict_users = pickle.load(open(pkl_path, 'rb'))
        else:
            num_shards = 100  # 150
            num_imgs = int(len(self.data) / num_shards)  # Image
            idx_shard = [i for i in range(num_shards)]  # shard idx list: [0, 1, 2, ......, 199]
            dict_users = {i: np.array([], dtype=int) for i in range(num_users)}  # user_idx dict
            idxs = np.arange(num_shards * num_imgs)  # img idx list: [0, 1, 2, ..., 144999], 145000 images
            img_idx = [i for i in range(len(self.data))]

            # divide and assign 2 shards/client
            for i in range(num_users):
                rand_set = set(np.random.choice(idx_shard, int(num_shards / num_users), replace=False))  # idx_shardnum_shards/num_users
                idx_shard = list(set(idx_shard) - rand_set)  # Shards
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs: (rand+1)*num_imgs]), axis=0)
                    img_idx = list(set(img_idx) - set(idxs[rand*num_imgs: (rand+1)*num_imgs]))

            dict_users[i] = np.concatenate([dict_users[i], img_idx])
            pickle.dump(dict_users, open(pkl_path, 'wb'))
        return dict_users

class F30kCaptionsCap(Dataset):
    def __init__(self, annFile='./dataset_k_split.pkl', train=True,
                 transform=None, target_transform=None, is_iid=False, client=-1):
        split = 'train' if train else 'test'
        self.transform = transform
        self.data = pickle.load(open(annFile, 'rb'))
        if split not in self.data.keys():
            assert False, f'split wrong {split}'
        self.data = self.data[split]  # 145,000 img-txt pairs, in list
            
        if client > -1 and train:
            # print(self.data)
            indices = self.iid()[client] if is_iid else self.non_iid()[client]
            indices = np.array(list(indices)).astype(int)

            self.data = [self.data[i] for i in indices]

        # self.data = [self.data[i] for i in range(1000)]

        images = [d[0] for d in self.data]
        self.n_images = len(set(images))
        # self.n_images = len(self.data) / 5
        self.iid_to_cls = {}

        # if not target_transform:
        #     vocab_path = '/root/MFL/coco_vocab.pkl'
        #     if isinstance(vocab_path, str):
        #         vocab = Vocabulary()
        #         vocab.load_from_pickle(vocab_path)
        #     else:
        #         vocab = vocab_path
        #     self.target_transform = caption_transform(vocab, 0)
        # else:
        #     self.target_transform = target_transform

    def iid(self, root='/autodl-fs/data/yClient/mmdata/Flickr30k/', num_users=20):
        """
        Sample I.I.D. client data from MNIST dataset
        :param dataset:
        :param num_users:
        :return: dict of image index
        """
        pkl_path = root + 'client_iid.pkl'
        if os.path.exists(pkl_path):
            dict_users = pickle.load(open(pkl_path, 'rb'))
        else:
            num_items = int(len(self.data) / num_users)
            dict_users, all_idxs = {}, [i for i in range(len(self.data))]
            for i in range(num_users):
                dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                     replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])
            pickle.dump(dict_users, open(pkl_path, 'wb'))
        return dict_users

    def non_iid(self, root='/root/MFL/data_partition/', num_users=10):
        pkl_path = root + 'client_noniid_flickr30k.pkl'
        if os.path.exists(pkl_path):
            dict_users = pickle.load(open(pkl_path, 'rb'))
        else:
            num_shards = 150  # 150
            num_imgs = int(len(self.data) / num_shards)  # Image
            idx_shard = [i for i in range(num_shards)]  # shard idx list: [0, 1, 2, ......, 199]
            dict_users = {i: np.array([], dtype=int) for i in range(num_users)}  # user_idx dict
            idxs = np.arange(num_shards * num_imgs)  # img idx list: [0, 1, 2, ..., 144999], 145000 images
            img_idx = [i for i in range(len(self.data))]

            # divide and assign 2 shards/client
            for i in range(num_users):
                rand_set = set(np.random.choice(idx_shard, int(num_shards / num_users), replace=False))  # idx_shardnum_shards/num_users
                idx_shard = list(set(idx_shard) - rand_set)  # Shards
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs: (rand+1)*num_imgs]), axis=0)
                    img_idx = list(set(img_idx) - set(idxs[rand*num_imgs: (rand+1)*num_imgs]))

            dict_users[i] = np.concatenate([dict_users[i], img_idx])
            pickle.dump(dict_users, open(pkl_path, 'wb'))
        return dict_users

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a caption for the annotation.
        """
        data = self.data[index]
        caption = data[1]

        path = data[0].replace('/data/mmdata/Flick30k/flickr30k-images/',
                              '/autodl-fs/data/yClient/mmdata/Flickr30k/flickr30k-images/')
        # print(f'path {path}')

        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     caption = self.target_transform(caption)

        return img, caption, index, int(index / 5)

    def __len__(self):
        return len(self.data)
    
    def get_image_indices(self):
        """
        Return a list of lists, where each sublist contains the indices of the same image.
        """
        image_to_indices = defaultdict(list)
        for idx, data in enumerate(self.data):
            image_path = data[0]
            image_to_indices[image_path].append(idx)
        return list(image_to_indices.values())

if __name__ == '__main__':
    train = F30kCaptionsCap(train=True, is_iid=False, client=1)
    print(len(train))
    print(train.n_images)
