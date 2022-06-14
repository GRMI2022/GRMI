import numpy as np
import torch

import pickle

from models.gnn_model_pom import Multmodality
from utils.utils import build_optimizer, get_known_mask


def train_gnn_y(data, args, log_path, device=torch.device('cpu')):
    model = Multmodality(data, args).to(device)
    model.load_state_dict(torch.load('./pom/test/pom_y_pretrain/model.pt'))
    trainable_parameters = list(model.parameters())

    scheduler, opt = build_optimizer(args, trainable_parameters)
    criterion = torch.nn.L1Loss(size_average=True)

    Train_loss = []
    Test_l1 = []
    Test_acc = []
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

    train_y_mask = all_train_y_mask.clone().detach()
    print("all y num is {}, train num is {}, test num is {}".format(
        all_train_y_mask.shape[0], torch.sum(train_y_mask),
        torch.sum(test_y_mask)))

    for epoch in range(args.epochs):
        model.train()
        audio_edge_num = edge_audio_attr.shape[0]
        visual_edge_num = edge_visual_attr.shape[0]
        text_edge_num = edge_text_attr.shape[0]
        audio_mask = get_known_mask(1 - args.known, int(audio_edge_num / 2)).to(device)
        visual_mask = get_known_mask(1 - args.known, int(visual_edge_num / 2)).to(device)
        text_mask = get_known_mask(1 - args.known, int(text_edge_num / 2)).to(device)

        known_train_edge_index = train_edge_index[:,
                                 ~torch.cat((audio_mask, audio_mask, visual_mask, visual_mask, text_mask, text_mask), dim=0)].clone()
        known_edge_audio_attr = edge_audio_attr[~torch.cat((audio_mask, audio_mask), dim=0)].clone()
        known_edge_visual_attr = edge_visual_attr[~torch.cat((visual_mask, visual_mask), dim=0)].clone()
        known_edge_text_attr = edge_text_attr[~torch.cat((text_mask, text_mask), dim=0)].clone()

        opt.zero_grad()

        edge_attr, node_embedding, edge_embedding, pred = model(x, known_edge_audio_attr, known_edge_visual_attr, known_edge_text_attr,
                                                                known_train_edge_index, edge_index)

        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]
        loss = criterion(pred_train, label_train)

        loss.backward()
        opt.step()
        train_loss = loss.item()
        Train_loss.append(train_loss)
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])

        model.eval()
        with torch.no_grad():
            edge_attr, node_embedding, edge_embedding, pred = model(x, edge_audio_attr, edge_visual_attr, edge_text_attr,
                                                                    train_edge_index, edge_index)

            pred_test = pred[test_y_mask]
            label_test = y[test_y_mask]
            test_acc = torch.mean(torch.mean((torch.round(pred_test) == torch.round(label_test)).float(), dim=0)).item()

            Test_acc.append(test_acc)
            if epoch % 100 == 0:
                print('epoch: ', epoch)
                print('loss: ', train_loss)

    pred_train = pred_train.detach().cpu().numpy()
    label_train = label_train.detach().cpu().numpy()
    pred_test = pred_test.detach().cpu().numpy()
    label_test = label_test.detach().cpu().numpy()

    obj = dict()
    obj['args'] = args
    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss

    obj['curves']['test_l1'] = Test_l1
    obj['curves']['test_acc'] = Test_acc
    obj['lr'] = Lr
    obj['outputs'] = dict()
    obj['outputs']['pred_train'] = pred_train
    obj['outputs']['label_train'] = label_train
    obj['outputs']['pred_test'] = pred_test
    obj['outputs']['label_test'] = label_test
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    print('test_acc:', np.max(obj['curves']['test_acc']))
