# -*- coding: utf-8 -*-

import os.path as osp
import inspect
from torch_geometric.data import Data
import torch
import numpy as np
from utils.utils import get_known_mask
from torch.utils.data import Dataset
import sys

if sys.version_info.major == 2:
    import cPickle as pickle
else:
    import pickle


AUDIO = b'covarep'
VISUAL = b'facet'
TEXT = b'glove'
LABEL = b'label'
TRAIN = b'train'
VALID = b'valid'
TEST = b'test'


def create_node(nrow, ncol, mode):
    if mode == 0:
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol, ncol))
        feature_node[np.arange(ncol), feature_ind] = 1
        sample_node = [[1] * ncol for i in range(nrow)]
        node = sample_node + feature_node.tolist()
    elif mode == 1:
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol, ncol + 1))
        feature_node[np.arange(ncol), feature_ind + 1] = 1
        sample_node = np.zeros((nrow, ncol + 1))
        sample_node[:, 0] = 1
        node = sample_node.tolist() + feature_node.tolist()
    return node


def create_edge(n_row, n_col):
    edge_start = []
    edge_end = []
    for x in range(n_row):
        edge_start = edge_start + [x] * n_col  # obj
        edge_end = edge_end + list(n_row + np.arange(n_col))  # att
    edge_start_new = edge_start + edge_end
    edge_end_new = edge_end + edge_start
    return (edge_start_new, edge_end_new)


def create_edge_attr(train_pom, val_pom, test_pom):
    edge_audio_attr = train_pom.audio.tolist() + val_pom.audio.tolist() + test_pom.audio.tolist()
    edge_visual_attr = train_pom.visual.tolist() + val_pom.visual.tolist() + test_pom.visual.tolist()
    edge_text_attr = train_pom.text.mean(axis=1).tolist() + val_pom.text.mean(axis=1).tolist() + test_pom.text.mean(axis=1).tolist()

    edge_audio_attr = edge_audio_attr + edge_audio_attr
    edge_visual_attr = edge_visual_attr + edge_visual_attr
    edge_text_attr = edge_text_attr + edge_text_attr
    return edge_audio_attr, edge_visual_attr, edge_text_attr


def get_data(train_set, valid_set, test_set, input_dims, node_mode, train_edge_prob, split_sample_ratio, split_by, seed=0, normalize=True):
    train_samples = train_set.__len__()
    valid_samples = valid_set.__len__()
    test_samples = test_set.__len__()
    n_samples = train_samples + valid_samples + test_samples
    n_modality = 3
    edge_start, edge_end = create_edge(n_samples, n_modality)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_audio_attr, edge_visual_attr, edge_text_attr = create_edge_attr(train_set, valid_set, test_set)
    edge_audio_attr = torch.tensor(edge_audio_attr, dtype=torch.float)
    edge_visual_attr = torch.tensor(edge_visual_attr, dtype=torch.float)
    edge_text_attr = torch.tensor(edge_text_attr, dtype=torch.float)
    node_init = create_node(n_samples, n_modality, node_mode)
    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(np.concatenate([train_set.labels, valid_set.labels, test_set.labels]), dtype=torch.float)
    train_edge_mask = get_known_mask(train_edge_prob, int(edge_index.shape[1] / 2))
    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)
    train_audio_edge_index = edge_index[:, ::3]
    train_visual_edge_index = edge_index[:, 1::3]
    train_text_edge_index = edge_index[:, 2::3]
    train_y_mask = torch.tensor([True for _ in range(train_samples)] + [False for _ in range(valid_samples + test_samples)], dtype=torch.bool)
    val_y_mask = torch.tensor([False for _ in range(train_samples)] + [True for _ in range(valid_samples)] + [False for _ in range(test_samples)],
                              dtype=torch.bool)
    test_y_mask = torch.tensor([False for _ in range(train_samples + valid_samples)] + [True for _ in range(test_samples)], dtype=torch.bool)

    train_edge_index = torch.cat([train_audio_edge_index[:, double_train_edge_mask[::3]],
                                  train_visual_edge_index[:, double_train_edge_mask[1::3]],
                                  train_text_edge_index[:, double_train_edge_mask[2::3]]], dim=1)

    # train_edge_index = torch.cat([train_audio_edge_index[:, double_train_edge_mask[::3]],
    #                               train_visual_edge_index[:, (~torch.cat([test_y_mask, test_y_mask])) | double_train_edge_mask[1::3]],
    #                               train_text_edge_index[:, double_train_edge_mask[2::3]]], dim=1)

    # edge_audio_attr_reconstruction = edge_audio_attr[~double_train_edge_mask[::3], :]
    # edge_visual_attr_reconstruction = edge_visual_attr[~double_train_edge_mask[1::3], :]
    # edge_text_attr_reconstruction = edge_text_attr[~double_train_edge_mask[2::3], :]

    edge_audio_attr = edge_audio_attr[double_train_edge_mask[::3], :]
    edge_visual_attr = edge_visual_attr[double_train_edge_mask[1::3], :]
    # edge_visual_attr = edge_visual_attr[(~torch.cat([test_y_mask, test_y_mask])) | double_train_edge_mask[1::3], :]
    edge_text_attr = edge_text_attr[double_train_edge_mask[2::3], :]
    data = Data(x=x, y=y,
                edge_audio_attr=edge_audio_attr, edge_visual_attr=edge_visual_attr, edge_text_attr=edge_text_attr,

                train_y_mask=train_y_mask, val_y_mask=val_y_mask, test_y_mask=test_y_mask,
                train_edge_index=train_edge_index, edge_index=edge_index, input_dims=input_dims
                )
    return data


def load_data(args):
    pom_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    train_set, valid_set, test_set, input_dims = load_pom(pom_path + '/raw_data/')
    data = get_data(train_set, valid_set, test_set, input_dims, args.node_mode, args.train_edge / 2 + 0.5, args.split_sample, args.split_by,
                    args.seed)
    return data


def load_pom(data_path):
    class POM(Dataset):
        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    if sys.version_info.major == 2:
        pom_data = pickle.load(open(data_path + "pom.pkl", 'rb'))
    else:
        pom_data = pickle.load(open(data_path + "pom.pkl", 'rb'), encoding='bytes')
    pom_train, pom_valid, pom_test = pom_data[TRAIN], pom_data[VALID], pom_data[TEST]

    train_audio, train_visual, train_text, train_labels \
        = pom_train[AUDIO], pom_train[VISUAL], pom_train[TEXT], pom_train[LABEL]
    valid_audio, valid_visual, valid_text, valid_labels \
        = pom_valid[AUDIO], pom_valid[VISUAL], pom_valid[TEXT], pom_valid[LABEL]
    test_audio, test_visual, test_text, test_labels \
        = pom_test[AUDIO], pom_test[VISUAL], pom_test[TEXT], pom_test[LABEL]

    train_set = POM(train_audio, train_visual, train_text, train_labels)
    valid_set = POM(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = POM(test_audio, test_visual, test_text, test_labels)

    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims
