# -*- coding: utf-8 -*-
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch

from pretraining.pretrain_pom import pretrain_pom
from pom.pom_subparser import add_pom_subparser
from utils.utils import auto_select_gpu
from pom.pom_data import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None, )
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None, )  # default to be all true
    parser.add_argument('--aggr', type=str, default='mean')
    parser.add_argument('--node_dim', type=int, default=32)
    parser.add_argument('--edge_dim', type=int, default=32)
    parser.add_argument('--edge_mode', type=int, default=1)
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--impute_hiddens', type=str, default='')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--impute_out_dim', type=int, default=32)
    parser.add_argument('--predict_hiddens', type=str, default='')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.7)
    parser.add_argument('--valid', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='pom_y_pretrain')
    subparsers = parser.add_subparsers()
    add_pom_subparser(subparsers)
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        device = torch.device('cuda:{}'.format(cuda))
    else:
        print('Using CPU')
        device = torch.device('cpu')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_data(args)

    log_path = './{}/test/{}/'.format(args.domain, args.log_dir)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)

    pretrain_pom(data, args, log_path, device)


if __name__ == '__main__':
    main()
