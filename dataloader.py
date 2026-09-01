from utils import *
import scipy
from torch import nn
import torch.nn.functional as F
import yaml
import torch


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


def get_networks(config, task):

    if task == 'ABIDE':
        pcorr, label = get_ABIDE()
    else:
        pcorr, label = get_ADNI()

    with open('node_clus_map.pickle', 'rb') as handle:
        node_clus_map = pickle.load(handle)

    node_to_com_label = torch.tensor(np.array(list(node_clus_map.values())), dtype=torch.long).to(device)

    pcorr_array = np.array(pcorr)

    dataset = torch.tensor(pcorr_array, dtype=torch.float32).to(device)

    A_node, A_com, C = create_adj_incidence_matrix(dataset, 116, config, node_to_com_label)

    return pcorr, A_node, A_com, C, label, node_to_com_label
