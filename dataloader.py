from utils import *
import scipy
from torch import nn
import torch.nn.functional as F
import yaml


def get_ADNI(task):
    filepath = r'D:\data\ADNI.mat'

    dataset = scipy.io.loadmat(filepath)

    pcorr_all = dataset['Network_Data']
    label_all = (dataset['label'] - 1).squeeze()

    if task == 'AD_NC':
        pcorr, label = pcorr_all['AD_NC_d'], label_all['AD_NC_d']
    elif task == 'EMCI_LMCI':
        pcorr, label = pcorr_all['EMCI_LMCI_d'], label_all['EMCI_LMCI_d']
    elif task == 'AD_LMCI':
        pcorr, label = pcorr_all['AD_LMCI_d'], label_all['AD_LMCI_d']

    return pcorr, label


def get_ABIDE():
    filepath = r'D:\data\ABIDE.mat'

    dataset = scipy.io.loadmat(filepath)

    pcorr = dataset['Network_Data']
    label = (dataset['label'] - 1).squeeze()


    return pcorr, label


def get_networks(dataset, task):
    if dataset == 'ABIDE':
        pcorr, label = get_ABIDE()
    else:
        pcorr, label = get_ADNI(task)

    return pcorr, label