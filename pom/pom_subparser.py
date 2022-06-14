# -*- coding: utf-8 -*-

def add_pom_subparser(subparsers):
    subparser = subparsers.add_parser('pom')
    subparser.add_argument('--domain', type=str, default='pom')
    subparser.add_argument('--train_edge', type=float, default=1)
    subparser.add_argument('--split_sample', type=float, default=0.)
    subparser.add_argument('--split_by', type=str, default='y')
    subparser.add_argument('--split_train', action='store_true', default=False)
    subparser.add_argument('--split_test', action='store_true', default=False)
    subparser.add_argument('--edge_input_dim', type=float, default=64)
    subparser.add_argument('--n_modalities', type=int, default=3)
    subparser.add_argument('--output_activation', type=str, default=None)
    subparser.add_argument('--predict_out_dim', type=int, default=16)
    subparser.add_argument('--node_mode', type=int, default=0)
