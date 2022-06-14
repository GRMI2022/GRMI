import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
from models.egcn import EGCNConv
from models.egsage import EGraphSage
from models.prediction_model import MLPNet
from utils.utils import get_activation


class Multmodality(torch.nn.Module):
    def __init__(self, data, args):
        super(Multmodality, self).__init__()

        self.n_row, self.n_col = data.y.shape[0], args.n_modalities
        self.GNN_model = get_gnn(data, args)
        self.inputLayer = InputNet(data.input_dims, args)
        if args.impute_hiddens == '':
            impute_hiddens = []
        else:
            impute_hiddens = list(map(int, args.impute_hiddens.split('_')))
        if args.concat_states:
            input_dim = args.node_dim * len(self.GNN_model.convs) * 2
        else:
            input_dim = args.node_dim * 2
        impute_out_dim = args.impute_out_dim
        self.impute_model = MLPNet(input_dim, impute_out_dim,
                                   hidden_layer_sizes=impute_hiddens,
                                   hidden_activation=args.impute_activation,
                                   dropout=args.dropout)

        if args.predict_hiddens == '':
            predict_hiddens = []
        else:
            predict_hiddens = list(map(int, args.predict_hiddens.split('_')))
        self.predict_model = MLPNet(args.node_dim, args.predict_out_dim,
                                    hidden_layer_sizes=predict_hiddens,
                                    output_activation=args.output_activation,
                                    dropout=args.dropout)

    def forward(self, x, audio_x, video_x, text_x, train_edge_index, edge_index):
        edge_attr = self.inputLayer(audio_x, video_x, text_x)
        node_embedding = self.GNN_model(x, edge_attr, train_edge_index)
        edge_embedding = self.impute_model(
            [node_embedding[edge_index[0, :int(self.n_row * self.n_col)]], node_embedding[edge_index[1, :int(self.n_row * self.n_col)]]])
        pred = self.predict_model(node_embedding[:-self.n_col])
        return edge_attr, node_embedding, edge_embedding, pred


def get_gnn(data, args):
    model_types = args.model_types.split('_')
    if args.norm_embs is None:
        norm_embs = [True, ] * len(model_types)
    else:
        norm_embs = list(map(bool, map(int, args.norm_embs.split('_'))))
    if args.post_hiddens is None:
        post_hiddens = [args.node_dim]
    else:
        post_hiddens = list(map(int, args.post_hiddens.split('_')))
    print(model_types, norm_embs, post_hiddens)
    model = GNNStack(data.num_node_features, args.edge_input_dim,
                     args.node_dim, args.edge_dim, args.edge_mode,
                     model_types, args.dropout, args.gnn_activation,
                     args.concat_states, post_hiddens,
                     norm_embs, args.aggr)
    return model


class GNNStack(torch.nn.Module):
    def __init__(self,
                 node_input_dim, edge_input_dim,
                 node_dim, edge_dim, edge_mode,
                 model_types, dropout, activation,
                 concat_states, node_post_mlp_hiddens,
                 normalize_embs, aggr
                 ):
        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.concat_states = concat_states
        self.model_types = model_types
        self.gnn_layer_num = len(model_types)
        self.convs = self.build_convs(node_input_dim, edge_input_dim,
                                      node_dim, edge_dim, edge_mode,
                                      model_types, normalize_embs, activation, aggr)
        if concat_states:
            self.node_post_mlp = self.build_node_post_mlp(int(node_dim * len(model_types)), int(node_dim * len(model_types)), node_post_mlp_hiddens,
                                                          dropout, activation)
        else:
            self.node_post_mlp = self.build_node_post_mlp(node_dim, node_dim, node_post_mlp_hiddens, dropout, activation)

        self.edge_update_mlps = self.build_edge_update_mlps(node_dim, edge_input_dim, edge_dim, self.gnn_layer_num, activation)

    def build_node_post_mlp(self, input_dim, output_dim, hidden_dims, dropout, activation):
        if 0 in hidden_dims:
            return get_activation('none')
        else:
            layers = []
            for hidden_dim in hidden_dims:
                layer = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    get_activation(activation),
                    nn.Dropout(dropout),
                )
                layers.append(layer)
                input_dim = hidden_dim
            layer = nn.Linear(input_dim, output_dim)
            layers.append(layer)
            return nn.Sequential(*layers)

    def build_convs(self, node_input_dim, edge_input_dim,
                    node_dim, edge_dim, edge_mode,
                    model_types, normalize_embs, activation, aggr):
        convs = nn.ModuleList()
        conv = self.build_conv_model(model_types[0], node_input_dim, node_dim,
                                     edge_input_dim, edge_mode, normalize_embs[0], activation, aggr)
        convs.append(conv)
        for l in range(1, len(model_types)):
            conv = self.build_conv_model(model_types[l], node_dim, node_dim,
                                         edge_dim, edge_mode, normalize_embs[l], activation, aggr)
            convs.append(conv)
        return convs

    def build_conv_model(self, model_type, node_in_dim, node_out_dim, edge_dim, edge_mode, normalize_emb, activation, aggr):
        if model_type == 'GCN':
            return pyg_nn.GCNConv(node_in_dim, node_out_dim)
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv(node_in_dim, node_out_dim)
        elif model_type == 'GAT':
            return pyg_nn.GATConv(node_in_dim, node_out_dim)
        elif model_type == 'EGCN':
            return EGCNConv(node_in_dim, node_out_dim, edge_dim, edge_mode)
        elif model_type == 'EGSAGE':
            return EGraphSage(node_in_dim, node_out_dim, edge_dim, activation, edge_mode, normalize_emb, aggr)

    def build_edge_update_mlps(self, node_dim, edge_input_dim, edge_dim, gnn_layer_num, activation):
        edge_update_mlps = nn.ModuleList()
        edge_update_mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim + edge_input_dim, edge_dim),
            get_activation(activation),
        )
        edge_update_mlps.append(edge_update_mlp)
        for l in range(1, gnn_layer_num):
            edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim + node_dim + edge_dim, edge_dim),
                get_activation(activation),
            )
            edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0], :]
        x_j = x[edge_index[1], :]
        edge_attr = mlp(torch.cat((x_i, x_j, edge_attr), dim=-1))
        return edge_attr

    def forward(self, x, edge_attr, edge_index):

        if self.concat_states:
            concat_x = []
        for l, (conv_name, conv) in enumerate(zip(self.model_types, self.convs)):
            if conv_name == 'EGCN' or conv_name == 'EGSAGE':
                x = conv(x, edge_attr, edge_index)
            else:
                x = conv(x, edge_index)
            if self.concat_states:
                concat_x.append(x)
            edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
        if self.concat_states:
            x = torch.cat(concat_x, 1)
        x = self.node_post_mlp(x)
        return x

    def check_input(self, xs, edge_attr, edge_index):
        Os = {}
        for indx in range(128):
            i = edge_index[0, indx].detach().numpy()
            j = edge_index[1, indx].detach().numpy()
            xi = xs[i].detach().numpy()
            xj = list(xs[j].detach().numpy())
            eij = list(edge_attr[indx].detach().numpy())
            if str(i) not in Os.keys():
                Os[str(i)] = {'x_j': [], 'e_ij': []}
            Os[str(i)]['x_i'] = xi
            Os[str(i)]['x_j'] += xj
            Os[str(i)]['e_ij'] += eij

        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1, 3, 1)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_i'], label=str(i))
            plt.title('x_i')
        plt.legend()
        plt.subplot(1, 3, 2)
        for i in Os.keys():
            plt.plot(Os[str(i)]['e_ij'], label=str(i))
            plt.title('e_ij')
        plt.legend()
        plt.subplot(1, 3, 3)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_j'], label=str(i))
            plt.title('x_j')
        plt.legend()
        plt.show()


class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class InputNet(nn.Module):
    def __init__(self, input_dims, args):
        super(InputNet, self).__init__()

        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]

        self.out_dim = args.edge_input_dim

        self.audio_prob = 0.2
        self.video_prob = 0.2
        self.text_prob = 0.2

        self.audio_subnet = SubNet(self.audio_in, self.out_dim, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.out_dim, self.video_prob)
        self.text_subnet = SubNet(self.text_in, self.out_dim, self.text_prob)

    def forward(self, audio_x, video_x, text_x):
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        return torch.cat([audio_h, video_h, text_h], dim=0)
