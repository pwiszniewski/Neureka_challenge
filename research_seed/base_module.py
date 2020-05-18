"""
This file defines the core research contribution
"""
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from argparse import ArgumentParser

import pytorch_lightning as pl
from utils.datasets import TUHDatasetNpy, TUHIterableDatasetNpy, TUHDatasetWithNamesAndTimeNpy

from sklearn.model_selection import StratifiedKFold

import numpy as np

import matplotlib.pyplot as plt

from utils.transforms import MTSA, FFTWithTimeFreqCorrelation
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing


class BaseModule(pl.LightningModule):

    def __init__(self, hparams):
        super(BaseModule, self).__init__()
        self.hparams = hparams
        self.num_workers = 3
        torch.random.manual_seed(0)
        input_shape = (20, 250)
        # input_shape = (20, 101)
        num_classes = 2
        self.init_model(input_shape, num_classes)
        self.ref_types = ['le']
        # self.ref_types = None
        self.is_undersamp = False
        # self.loss_name = 'cross_entropy'
        self.loss_name = 'mse_loss'

        if self.loss_name == 'cross_entropy':
            self.one_hot = False
        elif self.loss_name == 'mse_loss':
            self.one_hot = True

        self.split_dev = False
        self.is_iterable_dataloader = False

        self.is_norm = False
        self.scaler = preprocessing.StandardScaler()

        self.transform = None
        # self.transform = MTSA
        # self.transform = FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')


        self.test_results = []

    def prepare_data(self):
        self.dev_dataset = TUHDatasetNpy('dev', ref_types=self.ref_types, one_hot=self.one_hot, transform=self.transform)

        if self.split_dev:
            shuffle_dataset = False
            random_seed = 42
            skf = StratifiedKFold(n_splits=2, random_state=random_seed, shuffle=shuffle_dataset)
            val_indices, test_indices = list(skf.split(self.dev_dataset.data, self.dev_dataset.labels_org))[0]
            self.val_sampler = SubsetRandomSampler(val_indices)
            self.test_sampler = SubsetRandomSampler(test_indices)
        else:
            self.val_dataset = self.dev_dataset
            self.test_dataset = TUHDatasetWithNamesAndTimeNpy('dev', ref_types=self.ref_types, one_hot=self.one_hot,
                                                              transform=self.transform)

        if self.is_iterable_dataloader:
            self.train_dataset = TUHIterableDatasetNpy('train', ref_types=self.ref_types)
        else:
            self.train_dataset = TUHDatasetNpy('train', ref_types=self.ref_types, one_hot=self.one_hot,
                                               is_undersamp=self.is_undersamp, transform=self.transform)
            num_train_bckg = sum(self.train_dataset.labels_org == 0)
            num_train_seiz = sum(self.train_dataset.labels_org == 1)
            num_train = num_train_bckg + num_train_seiz
            self.class_weights = torch.tensor([num_train_seiz/num_train, num_train_bckg/num_train]).float().cuda()
            print('weights', self.class_weights)

        if self.is_norm:
            print('train')
            X_train_shape = self.train_dataset.data.shape
            self.train_dataset.data = self.train_dataset.data.reshape(self.train_dataset.data.shape[0], -1)
            self.train_dataset.data = self.scaler.fit_transform(self.train_dataset.data)
            self.train_dataset.data = self.train_dataset.data.reshape(X_train_shape)
            print('val')
            X_val_shape = self.val_dataset.data.shape
            self.val_dataset.data = self.val_dataset.data.reshape(self.val_dataset.data.shape[0], -1)
            self.val_dataset.data = self.scaler.fit_transform(self.val_dataset.data)
            self.val_dataset.data = self.val_dataset.data.reshape(X_val_shape)
        # self.class_weights = torch.FloatTensor([1/num_train_seiz, 1/num_train_bckg]).cuda()

    def forward(self, x):
        # x = torch.rand(x.size()[0], 20, 250).cuda()
        return self.net(x.float())

    def training_step(self, batch, batch_idx):
        x, y = batch
        # for i in range(x.size()[0]):
        #     plt.plot(x.cpu().detach().numpy()[i].T)
        #     plt.title(y[i].cpu().detach().numpy())
        #     plt.show()
        y_hat = self.forward(x)
        loss = self.calc_loss(y_hat, y)
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.calc_loss(y_hat, y)
        return {'val_loss': loss, 'val_out': y_hat.cpu().detach().numpy()}


    def test_step(self, batch, batch_idx):
        if self.split_dev:
            x, y = batch
            y_hat = self.forward(x)
            loss = self.calc_loss(y_hat, y)
            return {'test_loss': loss}
        else:
            fname, time_range, x, y = batch
            y_hat = self.forward(x)
            return {'fname': fname, 'time_range': time_range, 'test_out': y_hat}

    def calc_loss(self, y_hat, y):
        if self.loss_name == 'cross_entropy':
            return F.cross_entropy(y_hat, y)
        elif self.loss_name == 'mse_loss':
            return F.mse_loss(y_hat, y)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_outs = np.concatenate([x['val_out'] for x in outputs])
        # val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)

        if val_outs.shape[1] > 1:
            val_outs = np.argmax(val_outs, axis=1)
        y_cv = self.val_dataset.labels_org
        val_acc = sum(y_cv == val_outs) / (len(y_cv) * 1.0)
        f_score = f1_score(y_cv, val_outs, average='weighted')
        fpr, tpr, _ = roc_curve(val_outs, y_cv)
        # try:
        val_auc = auc(fpr, tpr)
        # except:
        #     val_auc = 0

        tensorboard_logs = {'loss/val': avg_loss, 'acc/val': val_acc, 'f_score/val': f_score, 'auc/val': val_auc}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        if self.split_dev:
            avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
            tensorboard_logs = {'test_loss': avg_loss}
            return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
        else:
            last_fname = None
            last_label = None
            last_start_time = 0
            last_stop_time = 0
            # results = []
            for batch_out in outputs:
                fnames, times, outs = batch_out.values()
                for i in range(len(fnames)):
                    fname = fnames[i]
                    time = times[i]
                    out = outs[i]
                    label = torch.argmax(out)
                    if fname != last_fname or label != last_label:
                        lbl_name = 'bckg' if last_label == 0 else 'seiz'
                        if last_label is not None:
                            self.test_results.append((last_fname, float(last_start_time), float(last_stop_time), lbl_name))
                        last_start_time = time[0]
                        last_label = label
                        last_fname = fname
                    last_stop_time = time[-1]
            return {'avg_test_loss': 'Unknown'}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        if self.split_dev:
            dataloader = DataLoader(self.dev_dataset, batch_size=self.hparams.batch_size, sampler=self.val_sampler,
                              num_workers=self.num_workers)
        else:
            dataloader = DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers)
        return dataloader

    def test_dataloader(self):
        if self.split_dev:
            dataloader = DataLoader(self.dev_dataset, batch_size=self.hparams.batch_size, sampler=self.test_sampler,
                              num_workers=self.num_workers)
        else:
            dataloader = DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers)
        return dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--batch_size', default=8192, type=int)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=20, type=int)
        # parser.add_argument('--num_workers', default=0, type=int)

        return parser