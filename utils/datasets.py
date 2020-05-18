from utils.inout import data_generator, DataGeneratorNpy, load_data_whole_ref_np

import numpy as np
from torch.utils.data.dataset import IterableDataset, Dataset
import copy
from itertools import tee
from itertools import cycle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def my_cycle(data_generator):
    iterator = data_generator.get_iterator()
    while True:
        for it in iterator:
            yield it

# def cycle(data_generator):
#     iterator = data_generator.get_iterator()
#     while True:
#         try:
#             yield next(iterator)
#         except StopIteration:
#             iterator = data_generator.get_iterator()
#             raise StopIteration

class TUHFeatIterableDataset(IterableDataset):
    def __init__(self, data_type='train'):
        win_len = 10
        self.data_generator = data_generator(win_len)

    def __iter__(self):
        return cycle(self.data_generator)


class TUHIterableDatasetNpy(IterableDataset):
    def __init__(self, data_type='train', ref_types=None):
        self.data_generator = DataGeneratorNpy(data_type=data_type, ref_types=ref_types)

    def __iter__(self):
        return self.data_generator.get_iterator()


class TUHIterableDatasetWithNamesAndTimeNpy(IterableDataset):
    def __init__(self, data_type='train', ref_types=None, one_hot=False):
        self.one_hot = one_hot
        self.data_generator = DataGeneratorNpy(data_type=data_type, ref_types=ref_types)

    def __iter__(self):
        return self.data_generator.get_iterator_with_names_and_time(onehot=self.one_hot)

    def __len__(self):
        return np.iinfo(np.int64).max
    #     return 10000

import multiprocessing
from multiprocessing import Pool, freeze_support
from itertools import repeat

def func(d, transform):
    # print(d.shape, transform, num)
    # return d.shape, transform, num
    return transform.apply(d)


def apply_transform(data, transform):
    freeze_support()
    with multiprocessing.Pool() as pool:
        res = pool.starmap(func, zip(data, repeat(transform)))
    return res

class TUHDatasetNpy(Dataset):
    def __init__(self, data_type='train', ref_types=None, one_hot=False, is_oversamp=False, is_undersamp=False,
                 transform=None):
        # gen = DataGeneratorNpy(data_type=data_type, ref_types=ref_types)
        # data_iter = gen.get_iterator()
        # self.data, self.labels = [], []
        # for d, l in data_iter:
        #     self.data.append(d)
        #     self.labels.append(l)
        # self.data = np.array(self.data)
        # self.len = len(self.labels)
        self.data = load_data_whole_ref_np(data_type=data_type, ref_types=ref_types)
        self.labels_org = np.any(self.data[:, 0, :], axis=1).astype(np.int)
        self.data = self.data[:, 1:]

        if is_oversamp:
            oversample = SMOTE()
            X, y = oversample.fit_resample(self.data, self.labels_org)
        if one_hot:
            lbls_onehot = convert_lbls_to_onehot(self.labels_org)
            self.labels = lbls_onehot.astype(np.float32)
        else:
            self.labels = self.labels_org

        if is_undersamp:
            # rus = RandomUnderSampler(random_state=0, replacement=False)
            # X_resampled, y_resampled = rus.fit_resample(self.data, self.labels_org)
            seiz_idxs = np.argwhere(self.labels_org == 1).ravel()
            bckg_idxs = np.random.choice(np.argwhere(self.labels_org == 0).ravel(), size=len(seiz_idxs))
            idxs = np.sort(np.concatenate([seiz_idxs, bckg_idxs]))
            self.data = self.data[idxs]
            self.labels = self.labels[idxs]
            self.labels_org = self.labels_org[idxs]

        if transform:
            self.data = apply_transform(self.data, transform)

        # # self.data_tranformed = np.zeros((self.data.shape[0], 21, 101))
        # if transform:
        #     # num = 10000
        #     # self.data = self.data[:num]
        #     # self.labels = self.labels[:num]
        #     self.data[:, :, :101] = transform().apply(self.data)
        #     # for i in range(self.data.shape[0]):
        #     #     self.data[i, :, :101] = transform().apply(self.data[i])
        #     #     pass
        #     self.data = self.data[:, :, :101]

        # self.data = self.data_tranformed

        self.len = self.labels.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.len

class TUHDatasetWithNamesAndTimeNpy(Dataset):
    def __init__(self, data_type='train', ref_types=None, one_hot=False, transform=None):
        self.one_hot = one_hot
        self.data_generator = DataGeneratorNpy(data_type=data_type, ref_types=ref_types)
        self.data = np.array(list(self.data_generator.get_iterator_with_names_and_time(onehot=self.one_hot)))
        if transform:
            self.data[:, 2] = apply_transform(self.data[:, 2], transform)
        # if transform:
        #     self.data = apply_transform(self.data, transform)
        self.len = len(self.data)

    def __getitem__(self, index):
        return tuple(self.data[index])

    def __len__(self):
        return self.len

def convert_lbls_to_onehot(labels_org):
    num_labels = 2
    lbls_onehot = np.zeros((labels_org.size, num_labels))
    lbls_onehot[np.arange(labels_org.size), labels_org] = 1
    return lbls_onehot