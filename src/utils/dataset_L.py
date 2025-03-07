import math
import numpy as np
import os
import pickle
import random
import torch
import torch
import torch
import torch.utils.data as data
import torchtext
from PIL import Image
from functools import partial
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchvision import transforms
from tqdm import tqdm

from src.utils.vocab import Vocabulary

def tokenize(sentence, vocab, caption_drop_prob):
    """nltk word_tokenize for caption transform.
    """
    tokens = word_tokenize(str(sentence).lower())
    tokenized_sentence = []
    tokenized_sentence.append(vocab('<start>'))
    tokenized = [vocab(token) for token in tokens]
    if caption_drop_prob > 0:
        unk = vocab('<unk>')
        tokenized = [vocab(token) if random.random() > caption_drop_prob else unk for token in tokens]
    else:
        tokenized = [vocab(token) for token in tokens]
    if caption_drop_prob:
        N = int(len(tokenized) * caption_drop_prob)
        for _ in range(N):
            tokenized.pop(random.randrange(len(tokenized)))
    tokenized_sentence.extend(tokenized)
    tokenized_sentence.append(vocab('<end>'))
    return torch.Tensor(tokenized_sentence)


def caption_transform(vocab, caption_drop_prob=0):
    """Transform for captions.
    "caption drop augmentation" randomly alters the given input tokens as <unk>
    """
    transform = []
    if caption_drop_prob < 0 or caption_drop_prob is None:
        print('warning: wrong caption drop prob', caption_drop_prob, 'set to zero')
        caption_drop_prob = 0
    elif caption_drop_prob > 0:
        print('adding caption drop prob', caption_drop_prob)
    transform.append(partial(tokenize, vocab=vocab, caption_drop_prob=caption_drop_prob))
    transform = transforms.Compose(transform)
    return transform


def text_cls(dset_name, istrain=False):
    if dset_name == 'AG_NEWS':
        # train: 120000， test: 7600， cls=4
        dset = torchtext.datasets.AG_NEWS(root=os.environ['HOME'] + '/autodl-tmp/shared-nvme',
                                          split='train' if istrain else 'test')
    elif dset_name == 'SogouNews':
        # train: 450000, test: 60000， cls=5
        dset = torchtext.datasets.SogouNews(root=os.environ['HOME'] + '/autodl-tmp/shared-nvme',
                                            split='train' if istrain else 'test')
    elif dset_name == 'DBpedia':
        # train: 560000, test: 70000， cls=14
        dset = torchtext.datasets.DBpedia(root=os.environ['HOME'] + '/autodl-tmp/shared-nvme',
                                          split='train' if istrain else 'test')
    elif dset_name == 'YelpReviewPolarity':
        # train: 560000, test: 38000， cls=2
        dset = torchtext.datasets.YelpReviewPolarity(root=os.environ['HOME'] + '/autodl-tmp/shared-nvme',
                                                     split='train' if istrain else 'test')
    elif dset_name == 'YelpReviewFull':
        # train: 650000, test: 50000， cls=5
        dset = torchtext.datasets.YelpReviewFull(root=os.environ['HOME'] + '/data',
                                                 split='train' if istrain else 'test')
    elif dset_name == 'YahooAnswers':
        # train: 1400000, test: 60000， cls=10
        dset = torchtext.datasets.YahooAnswers(root=os.environ['HOME'] + '/data',
                                               split='train' if istrain else 'test')
    elif dset_name == 'AmazonReviewPolarity':
        # train: 3600000, test: 400000， cls=2
        dset = torchtext.datasets.AmazonReviewPolarity(root=os.environ['HOME'] + '/data',
                                                       split='train' if istrain else 'test')
    elif dset_name == 'AmazonReviewFull':
        # train: 3000000, test: 650000， cls=5
        dset = torchtext.datasets.AmazonReviewFull(root=os.environ['HOME'] + '/data',
                                                   split='train' if istrain else 'test')
    elif dset_name == 'IMDB':
        # train: 25000, test: 25000， cls=2
        dset = torchtext.datasets.IMDB(root=os.environ['HOME'] + '/data', split='train' if istrain else 'test')
    return dset


def text_qa(dset_name, istrain=False):
    if dset_name == 'SQuAD1':
        # train: 87599， test: 10570
        dset = torchtext.datasets.SQuAD1(root=os.environ['HOME'] + '/data',
                                         split='train' if istrain else 'dev')
    elif dset_name == 'SQuAD2':
        # train: 130319, test: 11873
        dset = torchtext.datasets.SQuAD2(root=os.environ['HOME'] + '/data',
                                         split='train' if istrain else 'dev')
    return dset


def caption_collate_fn(data):
    """Build mini-batch tensors from a list of (image, sentence) tuples.
    Args:
      data: list of (image, sentence) tuple.
        - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
        - sentence: torch tensor of shape (?); variable length.

    Returns:
      targets: torch tensor of shape (batch_size, padded_length).
      lengths: list; valid length for each padded sentence.
    """
    # Sort a data list by sentence length
    # print(data[0])
    sentences = [i[0] for i in data]
    labels = [i[1] for i in data]
    labels = torch.Tensor(np.array(labels)).long()
    # print(f'sentences {sentences[0]}')
    # print(f'labels {labels[0]}')
    # print(f'data {data[0]}')

    # print('sen', len(sentences))
    # print('labels', labels)

    # data = [(i[1], i[0].long()) for i in data]

    # data = [(i[1], i[0].long()) for i in data]
    sentences.sort(key=lambda x: len(x), reverse=True)
    # labels, sentences = zip(*data)
    # for cap in sentences:
    #     print(cap, len(cap))

    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = [len(cap) for cap in sentences]
    targets = torch.zeros(len(sentences), max(cap_lengths)).long()
    # print(f'sentences {sentences}')
    for i, cap in enumerate(sentences):
        end = cap_lengths[i]
        targets[i, :end] = cap[:end]

    cap_lengths = torch.Tensor(cap_lengths).long()
    # print(labels)
    return targets, labels, cap_lengths


class Language(data.Dataset):

    def __init__(self, name='AG_NEWS', train=True, transform=None, is_iid=False,
                 client=-1, root=os.environ['HOME'] + '/autodl-tmp/shared-nvme/'):
        try:
            dataset = text_cls('AG_NEWS', istrain=train)
        except Exception as e:
            print(f"加载数据集时出错: {e}")
        dataset = enumerate(dataset)
        self.targets = []
        self.data = []

        for _, (l, t) in dataset:  # len: 7600
            self.targets.append(l)  # label: {1, 2, 3, 4} 4
            self.data.append(t)  # sentence: raw sentence
        self.targets = np.array(self.targets)
        self.targets -= min(self.targets)  # label: {0, 1, 2, 3}  4
        self.targets = np.array(self.targets)
        
        # if not transform:
        #     vocab_path = '/root/MFL/coco_vocab.pkl'
        #     if isinstance(vocab_path, str):
        #         vocab = Vocabulary()
        #         vocab.load_from_pickle(vocab_path)
        #     else:
        #         vocab = vocab_path
        #     transform = caption_transform(vocab, 0)

        # self.transform = transform

    def __getitem__(self, index):

        output = self.data[index]

        # if self.transform is not None:
        #     output = self.transform(self.data[index]).long()

        return output, self.targets[index]

    def __len__(self):
        return len(self.targets)