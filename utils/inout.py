import sys
import os

import pandas as pd
import numpy as np
import pyedflib
from utils.info import get_data_info
from settings import *

from utils.nedc import nedc_pystream
# from utils.nedc.nedc_print_labels import *

import utils.nedc.sys_tools.nedc_cmdl_parser as ncp
import utils.nedc.sys_tools.nedc_file_tools as nft
import utils.nedc.sys_tools.nedc_ann_tools as nat

import struct

import mne


def data_generator(length=10):
    files_info = get_data_info()
    tse_path = None
    for index, file_info in files_info.iterrows():
        path = DATA_DIR + file_info['Filename'][1:]
        if path != tse_path:
            tse_path = DATA_DIR + file_info['Filename'][1:]
            raw_path = FEATS_DIR + file_info['Filename'][1:][:-3] + 'raw'
            features = load_feat_from_raw(raw_path)
            annotations = get_annotations(tse_path)
            labels = labels_from_annotations(annotations)
            print(index)
            for i in range(0, features.shape[0], length):
                yield features[i:i+length], int(labels[i:i+length].any())


# def data_generator_npy(chunk_len=250, data_type='train', dirs=None):
#     if dirs is None:
#         dirs = [os.path.join(NP_DATA_DIR, data_type, d) for d in os.listdir(os.path.join(NP_DATA_DIR, data_type))]
#     else:
#         dirs = [os.path.join(NP_DATA_DIR, data_type, d) for d in dirs]
#     for dir in dirs:
#         for file in os.listdir(dir):
#             data = np.load(os.path.join(dir, file))
#             for i in range(0, data.shape[1], chunk_len):
#                 yield data[1:, i:i+chunk_len], int(data[0, i:i+chunk_len].any())

class DataGeneratorNpy:
    def __init__(self, chunk_len=250, data_type='train', ref_types=None):
        self.chunk_len = chunk_len
        self.fs = 250
        if ref_types is None:
            self.dirs = [os.path.join(NP_DATA_DIR, data_type, d) for d in ref_types_to_dirs(['ar', 'le', 'ar_a'])]
        else:
            dirs = ref_types_to_dirs(ref_types)
            self.dirs = [os.path.join(NP_DATA_DIR, data_type, d) for d in dirs]

    def get_iterator(self):
        for dir in self.dirs:
            for file in os.listdir(dir):
                data = np.load(os.path.join(dir, file))
                for i in range(0, data.shape[1], self.chunk_len):
                    yield data[1:, i:i+self.chunk_len], int(data[0, i:i+self.chunk_len].any())

    def get_iterator_whole_arr(self):
        for dir in self.dirs:
            for file in os.listdir(dir):
                data = np.load(os.path.join(dir, file))
                for i in range(0, data.shape[1], self.chunk_len):
                    yield data[:, i:i+self.chunk_len]

    def get_iterator_with_names_and_time(self, onehot=False):
        for dir in self.dirs:
            for file in os.listdir(dir):
                data = np.load(os.path.join(dir, file))
                for i in range(0, data.shape[1], self.chunk_len):
                    label = int(data[0, i:i+self.chunk_len].any())
                    if onehot:
                        label = self._convert_label_to_onehot(label)
                    time_range = np.arange(i, i+self.chunk_len) / self.fs + 1 / self.fs
                    # time_range = (1/self.fs, self.chunk_len/self.fs)
                    yield file[:-4], time_range, data[1:, i:i+self.chunk_len], label

    def _convert_label_to_onehot(self, labels_org):
        num_labels = 2
        lbls_onehot = np.zeros((1, num_labels))
        lbls_onehot[np.arange(1), labels_org] = 1
        return lbls_onehot


def filter_ref_file_and_save_specific(ref_filepath, data_type='dev'):
    import re
    with open(ref_filepath) as ref_file:  # Use file to refer to the file object
        ref_data = ref_file.read().split("\n")
        # ref_dir, _ = os.path.split(ref_filepath)
        for ref in ['ar', 'le', 'ar_a']:
            print(ref)
            dir = os.path.join(NP_DATA_DIR, data_type, ref_types_to_dirs([ref])[0])
            idxs = []
            for file in os.listdir(dir):
                for i, item in enumerate(ref_data):
                    if file[:-4] in item:
                        idxs.append(i)
            idxs = np.sort(idxs)
            with open(ref_filepath[:-4]+'_'+ref+'.txt', "w") as fw:
                for i in idxs:
                    fw.write(ref_data[i]+'\n')
            print('')

            # for file in os.listdir(dir):
            #     file = '00000258_s002_t002'
            #     occur = [m.start() for m in re.finditer(file, ref_data)]
                # occur = [m.start() for m in re.finditer(file[:-4], ref_data)]


def load_npy_files_save_one(data_type='train', prec=np.float16):
    refs = ['ar', 'le']
    for ref, dir in zip(refs, ref_types_to_dirs(refs)):
        gen = DataGeneratorNpy(data_type=data_type, ref_types=[ref])
        data_iter = gen.get_iterator_whole_arr()
        data = []
        import matplotlib.pyplot as plt
        for d in data_iter:
            # plt.plot(d[1])
            # plt.plot(d.astype(np.byte)[1])
            data.append(d.astype(prec))
            # plt.show()
        data = np.array(data)
        np_path = os.path.join(NP_DATA_DIR, data_type, dir + '.npy')
        np.save(np_path, data)
        print('saved array', np_path)



def load_data_all():
    files_info = get_data_info()
    tse_path = None
    for index, file_info in files_info.iterrows():
        path = DATA_DIR + file_info['Filename'][1:]
        if path != tse_path:
            tse_path = DATA_DIR + file_info['Filename'][1:]
            raw_path = FEATS_DIR + file_info['Filename'][1:][:-3] + 'raw'
            edf_path = tse_path[:-3] + 'edf'
            features = load_feat_from_raw(raw_path)
            fsamp, sig, labels = load_data(edf_path)
            annotations = get_annotations(tse_path)
            print(index)
            # print(index, file_info['Patient'], features.shape, annotations)


def load_feats_and_save_numpy():
    files_info = get_data_info()
    tse_path = None
    for index, file_info in files_info.iterrows():
        path = DATA_DIR + file_info['Filename'][1:]
        if path != tse_path:
            tse_path = DATA_DIR + file_info['Filename'][1:]
            raw_path = FEATS_DIR + file_info['Filename'][1:][:-3] + 'raw'
            edf_path = tse_path[:-3] + 'edf'
            # num_frames, num_chann, num_feat = get_feat_raw_header(raw_path)
            params = load_params_for_edf(edf_path)
            fsamp, sig, labels = load_data(edf_path)


def load_edfs_resamp_and_save_numpy(f_resamp=250, data_type='train'):
    files_info = get_data_info(data_type)
    tse_path = None
    for index, file_info in files_info.iterrows():
        path = DATA_DIR + file_info['Filename'][1:]
        if path != tse_path:
            tse_path = DATA_DIR + file_info['Filename'][1:]
            edf_path = tse_path[:-3] + 'edf'
            _, filename = os.path.split(edf_path)
            np_path = NP_DATA_DIR + '/' + data_type + '/' + edf_path.split('/')[-5] + '/' + filename[:-3] + 'npy'
            fsamp, sig, labels = load_data(edf_path)
            np_dir, _ = os.path.split(np_path)
            os.makedirs(np_dir, exist_ok=True)
            annotations = get_annotations(tse_path)
            convert_time_annot_to_samples(annotations, f_resamp)
            labels = labels_from_annotations(annotations)
            if fsamp[0] != f_resamp:
                sig = mne.filter.resample(sig, up=f_resamp, down=fsamp[0])
            # if np.sum(labels) > 0:
            #     import matplotlib.pyplot as plt
            #     fig, ax = plt.subplots()
            #     x = np.linspace(0, len(out[1]), len(out[1]))
            #     ax.plot(x, out[1])
            #     ax.fill_between(x, 0, 1, where=out[0], color='green', alpha=0.5, transform=ax.get_xaxis_transform())
            #     plt.show()
            np.save(np_path, np.vstack([labels, sig]))
            print(index)


def load_data(edf_path):
    params = load_params_for_edf(edf_path)
    fsamp, sig, labels = nedc_pystream.nedc_load_edf(edf_path)
    fsamp_sel, sig_sel, labels_sel = nedc_pystream.nedc_select_channels(params, fsamp, sig, labels)
    fsamp_mont, sig_mont, labels_mont = nedc_pystream.nedc_apply_montage(params, fsamp_sel, sig_sel, labels_sel)
    return fsamp_mont, np.array(sig_mont), labels_mont


def load_data_whole_ref_np(data_type='train', ref_types=None):
    if ref_types is None:
        ref_types = ['ar', 'le', 'ar_a']
    dirs = ref_types_to_dirs(ref_types)
    if len(dirs) == 1:
        np_path = os.path.join(NP_DATA_DIR, data_type, dirs[0] + '.npy')
        data = np.load(np_path)
        return data
    else:
        data = []
        for d in dirs:
            np_path = os.path.join(NP_DATA_DIR, data_type, d + '.npy')
            data.append(np.load(np_path))
        return np.vstack(data)


def load_params_for_edf(edf_path, common=True):
    if '01_tcp_ar' in edf_path:
        params_path = PARAMS_TCP_AR_COMM_PATH if common else PARAMS_TCP_AR_PATH
    elif '02_tcp_le' in edf_path:
        params_path = PARAMS_TCP_LE_COMM_PATH if common else PARAMS_TCP_LE_PATH
    elif '03_tcp_ar_a' in edf_path:
        params_path = PARAMS_TCP_AR_A_COMM_PATH if common else PARAMS_TCP_AR_A_PATH
    else:
        print('Not found params for', edf_path)
        raise
    params = nedc_pystream.nedc_load_parameters(params_path)
    if params == None:
        print("(%s: %s) error loading parameters" % (sys.argv[0], __name__))
        exit(-1)
    return params


def get_annotations(label_path):
    lev = int(0)
    sub = int(0)
    ann = nat.Annotations()
    status = ann.load(nft.get_fullpath(label_path))
    if not status:
        print('Label file not found ', label_path)
        raise
    channel = -1
    annotations = ann.get(lev, sub, channel)
    return annotations

def get_feat_raw_header(path):
    with open(path, "rb") as file:
        INT_SIZE = 4
        FLOAT_SIZE = 4
        ORDER = 'little'
        b = file.read(INT_SIZE)
        num_chann = int.from_bytes(b, ORDER)
        b = file.read(INT_SIZE)
        num_frames = int.from_bytes(b, 'little')
        b = file.read(INT_SIZE)
        num_feat = int.from_bytes(b, 'little')
    return num_frames, num_chann, num_feat


def load_feat_from_raw(path):
    with open(path, "rb") as file:
        INT_SIZE = 4
        FLOAT_SIZE = 4
        ORDER = 'little'
        b = file.read(INT_SIZE)
        num_chann = int.from_bytes(b, ORDER)
        b = file.read(INT_SIZE)
        num_frames = int.from_bytes(b, 'little')

        feat_all_chann = []
        for ch in range(num_chann):
            feat = []
            for i in range(num_frames):
                b = file.read(INT_SIZE)
                num_feat = int.from_bytes(b, 'little')
                b = file.read(FLOAT_SIZE*num_feat)
                feat.append(struct.unpack(str(num_feat) + 'f', b))
            feat_all_chann.append(feat)
    return np.array(feat_all_chann).transpose(1, 0, 2)


def labels_from_annotations(annotations):
    labels = np.zeros(int(annotations[-1][1]))
    for ann in annotations:
        if list(ann[2])[0] != 'bckg':
            labels[int(ann[0]):int(ann[1])] = 1
    return labels


def convert_time_annot_to_samples(annot, fsamp):
    for ann in annot:
        ann[0] *= fsamp
        ann[1] *= fsamp


def ref_types_to_dirs(ref_types):
    dirs = []
    if any(x not in ['ar', 'le', 'ar_a'] for x in ref_types):
        print('not found one ref in', ref_types)
        return None
    if 'ar' in ref_types:
        dirs.append('01_tcp_ar')
    if 'le' in ref_types:
        dirs.append('02_tcp_le')
    if 'ar_a' in ref_types:
        dirs.append('03_tcp_ar_a')
    return dirs


def save_results(path, results):
    f = open(path, 'w')
    for res in results:
        f.write(f'{res[0]} {res[1]:.4f} {res[2]:.4f} {res[3]} 1.0000\n')
    f.close()