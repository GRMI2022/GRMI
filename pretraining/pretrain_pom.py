# -*- coding: utf-8 -*-
import torch

import torch.nn.functional as F
import pickle

from models.gnn_model_pom import Multmodality
from models.prediction_model import MLPNet
from utils.utils import build_optimizer, get_known_mask, cal_contrastive_score, infoNCELoss


def pretrain_pom(data, args, log_path, device=torch.device('cpu')):
    model = Multmodality(data, args).to(device)

    n_row, n_col = data.y.shape[0], args.n_modalities
    impute_out_dim = args.impute_out_dim
    edge_dim = args.edge_dim
    node_dim = args.node_dim
    masked_edge_reconstruction_model = MLPNet(impute_out_dim, args.edge_input_dim,
                                              hidden_layer_sizes=[],
                                              dropout=args.dropout).to(device)

    node_edge_matching_model = MLPNet(edge_dim + node_dim, 2,
                                      hidden_layer_sizes=[],
                                      dropout=args.dropout).to(device)
    trainable_parameters = list(model.parameters()) + list(masked_edge_reconstruction_model.parameters())

    scheduler, opt = build_optimizer(args, trainable_parameters)
    Train_loss = []
    Test_mse = []
    Lr = []

    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    edge_index = data.edge_index.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    edge_audio_attr = data.edge_audio_attr.clone().detach().to(device)
    edge_visual_attr = data.edge_visual_attr.clone().detach().to(device)
    edge_text_attr = data.edge_text_attr.clone().detach().to(device)
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)

    eye = torch.eye(n_row).to(device)

    train_y_mask = all_train_y_mask.clone().detach()
    print("all y num is {}, train num is {}, test num is {}".format(
        all_train_y_mask.shape[0], torch.sum(train_y_mask),
        torch.sum(test_y_mask)))

    for epoch in range(args.epochs):
        if epoch % 100 == 0:
            print('epoch:', epoch)
        model.train()
        masked_edge_reconstruction_model.train()

        audio_edge_num = edge_audio_attr.shape[0]
        visual_edge_num = edge_visual_attr.shape[0]
        text_edge_num = edge_text_attr.shape[0]
        audio_mask = get_known_mask(1 - args.known, int(audio_edge_num / 2)).to(device)
        visual_mask = get_known_mask(1 - args.known, int(visual_edge_num / 2)).to(device)
        text_mask = get_known_mask(1 - args.known, int(text_edge_num / 2)).to(device)
        all_mask = torch.cat((audio_mask, audio_mask, visual_mask, visual_mask, text_mask, text_mask), dim=0)
        known_train_edge_index = train_edge_index[:, ~all_mask].clone()
        known_edge_audio_attr = edge_audio_attr[~torch.cat((audio_mask, audio_mask), dim=0)].clone()
        known_edge_visual_attr = edge_visual_attr[~torch.cat((visual_mask, visual_mask), dim=0)].clone()
        known_edge_text_attr = edge_text_attr[~torch.cat((text_mask, text_mask), dim=0)].clone()

        opt.zero_grad()
        edge_attr, node_embedding, edge_embedding, pred = model(x, known_edge_audio_attr, known_edge_visual_attr, known_edge_text_attr,
                                                                known_train_edge_index, edge_index)
        masked_train_edge_index = train_edge_index[:, all_mask].clone()
        masked_train_edge_index = masked_train_edge_index[:, masked_train_edge_index[0] < n_row]
        X_reconstrution = masked_edge_reconstruction_model(edge_embedding)
        X_reconstrution = X_reconstrution[masked_train_edge_index[0] * 3 + (masked_train_edge_index[1] - n_row)]
        X_input = model.inputLayer(edge_audio_attr[:audio_edge_num // 2][audio_mask],
                                   edge_visual_attr[:visual_edge_num // 2][visual_mask],
                                   edge_text_attr[:text_edge_num // 2][text_mask])
        loss1 = F.mse_loss(X_input, X_reconstrution, reduction='sum')

        edge_embedding = torch.reshape(edge_embedding, [n_row, n_col, -1])
        contrastive_loss = []
        for i in range(n_col):
            for j in range(n_col):
                if j != i:
                    contrastive_loss.append(infoNCELoss(cal_contrastive_score(edge_embedding[:, i, :], edge_embedding[:, j, :])))

        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]

        loss2 = sum(contrastive_loss) / len(contrastive_loss)

        for i in range(n_col):
            sim = cal_contrastive_score(edge_embedding[:, i, :], edge_embedding[:, i, :])
            sim = sim - eye
            neg_sample_index = torch.argmax(sim, dim=0)
            pos_sample = torch.cat([node_embedding[:n_row], edge_embedding[:, i, :]], dim=-1)
            neg_sample = torch.cat([node_embedding[:n_row], edge_embedding[neg_sample_index, i, :]], dim=-1)
            matching_label = torch.tensor([1 for _ in range(n_row)] + [0 for _ in range(n_row)], dtype=torch.int64).to(device)
            pred = node_edge_matching_model(torch.cat([pos_sample, neg_sample], dim=0))
            loss3 = F.cross_entropy(pred, matching_label)

        loss = loss1 + loss2 + loss3
        loss.backward()
        opt.step()
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])

        model.eval()
        masked_edge_reconstruction_model.eval()
        with torch.no_grad():

            edge_attr, node_embedding, edge_embedding, pred = model(x, edge_audio_attr, edge_visual_attr, edge_text_attr, train_edge_index,
                                                                    edge_index)
            test_edge_index = train_edge_index[:, ~all_mask].clone()
            test_edge_index = test_edge_index[:, test_edge_index[0] < n_row]
            X_reconstrution = masked_edge_reconstruction_model(edge_embedding)
            X_reconstrution = X_reconstrution[test_edge_index[0] * 3 + (test_edge_index[1] - n_row)]
            X_input = model.inputLayer(edge_audio_attr[:audio_edge_num // 2][~audio_mask],
                                       edge_visual_attr[:visual_edge_num // 2][~visual_mask],
                                       edge_text_attr[:text_edge_num // 2][~text_mask])
            test_mse = F.mse_loss(X_input, X_reconstrution, reduction='sum')

            Train_loss.append(train_loss)
            Test_mse.append(test_mse.item())

    pred_train = pred_train.detach().cpu().numpy()
    label_train = label_train.detach().cpu().numpy()

    obj = dict()
    obj['args'] = args
    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss
    obj['curves']['test_mes'] = Test_mse

    obj['lr'] = Lr
    obj['outputs'] = dict()
    obj['outputs']['pred_train'] = pred_train
    obj['outputs']['label_train'] = label_train

    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    torch.save(model.state_dict(), log_path + 'model.pt')
    print('model saved')
