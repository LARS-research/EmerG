import time
import codecs
import csv
import os
import copy
import matplotlib.pyplot as plt
#import seaborn as sns
import torch
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import argparse
import nni
from data import MovieLens1MColdStartDataLoader, TaobaoADColdStartDataLoader
from model import EmerG
from model.gnn import GNN

from copy import deepcopy
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
plt.rcParams.update({
    "font.size": 17,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
    "font.sans-serif": ["Helvetica"]})

graph_num = 100

def generate_picture(A, B, item_id, layer_index, phase, year, title, genres, count):
    global graph_num
    layer_index = 2
    if graph_num > 0:
        features = ['user ID', 'gender', 'age', 'occupation', 'zip-code', 'item ID', 'year', 'title', 'genres']

        fig, ax = plt.subplots()
        im = ax.imshow(B, cmap='YlGnBu')
        ax.set_xticks(np.arange(len(features)))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels(features)
        ax.set_yticklabels(features)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")



        fig.tight_layout()
        plt.colorbar(im)
        plt.savefig(f'./graph/graph_{item_id}_{layer_index}_{phase}.png',dpi=300)
        plt.close('all')
        graph_num = graph_num - 1
    return

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model_path', default='')
    parser.add_argument('--dataset_name', default='taobaoAD', help='required to be one of [movielens1M, taobaoAD, baiduAD]')
    parser.add_argument('--datahub_path', default='./datahub/')
    parser.add_argument('--warmup_model', default='cvar', help="required to be one of [base, mwuf, metaE, cvar, cvar_init, melu, gme]")
    parser.add_argument('--is_dropoutnet', type=bool, default=False, help="whether to use dropout net for pretrain")
    parser.add_argument('--with_graph', type=bool, default=False, help="whether to use dropout net for pretrain")
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--model_name', default='deepfm', help='backbone name, we implemented [fm, wd, deepfm, afn, ipnn, opnn, afm, dcn, autoint, fignn]')
    parser.add_argument('--epoch', type=int, default=32)
    parser.add_argument('--pretrain_epochs', type=int, default=32)
    parser.add_argument('--melu_epochs', type=int, default=2)
    parser.add_argument('--cvar_epochs', type=int, default=2)
    parser.add_argument('--cvar_iters', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_inner', type=float, default=1e-3)
    parser.add_argument('--lr_pretrain', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--item_id_name', default='item_id')
    parser.add_argument('--runs', type=int, default=3, help = 'number of executions to compute the average metrics')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--meta_lr', type=float, default=1e-3)
    parser.add_argument('--warm_lr', type=float, default=1e-3)

    args = parser.parse_args()
    return args

def get_loaders(name, datahub_path, device, bsz, shuffle, maml_episode=False):
    path = os.path.join(datahub_path, name, "{}_data.pkl".format(name))
    if name == 'movielens1M':
        dataloaders = MovieLens1MColdStartDataLoader(name, path, device, bsz=bsz, shuffle=shuffle, maml_episode=maml_episode)
    elif name == 'taobaoAD':
        dataloaders = TaobaoADColdStartDataLoader(name, path, device, bsz=bsz, shuffle=shuffle)
    else:
        raise ValueError('unkown dataset name: {}'.format(name))
    return dataloaders

def get_model(name, dl):
    if name == 'gnn':
        return GNN(dl.description, embed_dim=16, gnn_layers=3, device=args.device)
    return

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def test(model, data_loader, device):
    model.eval()
    labels, scores, losses = list(), list(), list()
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        old_time_all = time.time()
        p = 0
        for _, (features, label) in enumerate(data_loader):
            if p == 0:
                old_time = time.time()
            features = {key: value.to(device) for key, value in features.items()}
            label = label.to(device)
            y = model(features)
            labels.extend(label.tolist())
            scores.extend(y.squeeze().tolist())
            losses.append(criterion(y.view(-1, 1), label.view(-1, 1).float()).item())
            
        
    scores_arr = np.array(scores)
    return np.mean(losses), roc_auc_score(labels, scores), f1_score(labels, (scores_arr > np.mean(scores_arr)).astype(np.float32).tolist())


def dropoutNet_train(model, data_loader, device, epoch, lr, weight_decay, save_path, log_interval=10, val_data_loader=None, loss_save_path=None):
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)
    loss_list = []
    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(data_loader) 
        for i, (features, label) in enumerate(data_loader):
            if random.random() < 0.1:
                bsz = label.shape[0]
                item_emb = model.emb_layer['item_id']
                origin_item_emb = item_emb(features['item_id']).squeeze(1)
                mean_item_emb = torch.mean(item_emb.weight.data, dim=0, keepdims=True) \
                                    .repeat(bsz, 1)
                y = model.forward_with_item_id_emb(mean_item_emb, features)
            else:
                y = model(features)
            loss = criterion(y, label.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                print("    iters {}/{} loss: {:.4f}".format(i + 1, total_iters + 1, total_loss/log_interval), end='\r')
                loss_list.append(total_loss/log_interval)
                total_loss = 0
        print("Epoch {}/{} loss: {:.4f}".format(epoch_i, epoch, epoch_loss/total_iters), " " * 20)
    file_loss = codecs.open(loss_save_path, 'w', 'utf-8')
    writer = csv.writer(file_loss)
    for i in loss_list:
        writer.writerow([i])
    return 

def train(model, data_loader, device, epoch, lr, weight_decay, save_path, log_interval=10, val_data_loader=None, loss_save_path=None):
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                          lr=lr, weight_decay=weight_decay)
    loss_list = []
    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(data_loader) 
        for i, (features, label) in enumerate(data_loader):
            y = model(features)
            loss = criterion(y.view(-1, 1), label.view(-1, 1).float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                print("    Epoch: {} Iter {}/{} loss: {:.4f}".format(epoch_i, i + 1, total_iters + 1, total_loss/log_interval), end='\r')
                loss_list.append(total_loss/log_interval)
                total_loss = 0
    if loss_save_path is not None:
        file_loss = codecs.open(loss_save_path, 'w', 'utf-8')
        writer = csv.writer(file_loss)
        for i in loss_list:
            writer.writerow([i])
    return 

def train_emb(model, data_loader, device, epoch, lr, weight_decay, save_path, log_interval=10, val_data_loader=None, loss_save_path=None):
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                          lr=lr, weight_decay=weight_decay)
    loss_list = []
    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(data_loader) 
        for i, x_dict in enumerate(data_loader):
            y, _, _ = model.forward_for_pretrain(x_dict)
            loss = criterion(y.view(-1, 1), torch.ones((y.size()[0], 1)).to(device))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                print("    Epoch: {} Iter {}/{} loss: {:.4f}".format(epoch_i, i + 1, total_iters + 1, total_loss/log_interval), end='\r')
                loss_list.append(total_loss/log_interval)
                total_loss = 0
    if loss_save_path is not None:
        file_loss = codecs.open(loss_save_path, 'w', 'utf-8')
        writer = csv.writer(file_loss)
        for i in loss_list:
            writer.writerow([i])
    return 


def pretrain(dataset_name, 
         datahub_name,
         bsz,
         shuffle,
         model_name,
         epoch,
         lr,
         weight_decay,
         seed,
         device,
         save_dir,
         is_dropoutnet=False,
         save_loss=True):
    device = torch.device(device)
    if 'fignn' in model_name:
        save_dir = os.path.join(save_dir, 'fignn')
    else:
        save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataloaders = get_loaders(dataset_name, datahub_name, device, bsz, shuffle==1)
    
    model = get_model(model_name, dataloaders).to(device)
    dropout_net_prefix = 'isdropout_' if is_dropoutnet else ''
    save_path = os.path.join(save_dir, f'{dataset_name}_{epoch}_{lr}_{weight_decay}_{seed}_{dropout_net_prefix}model_32.pth')
    if save_loss:
        loss_save_path = os.path.join(save_dir, f'{dataset_name}_{epoch}_{lr}_{weight_decay}_{seed}_{dropout_net_prefix}model_pretrain_32.csv')
    else:
        loss_save_path = None
    print('save model in', save_path)

    print("="*20, 'pretrain {}'.format(model_name), "="*20)
    # init parameters
    model.init()
    
    # pretrain
    pretrain_begin_time = time.time()
    if is_dropoutnet:
        dropoutNet_train(model, dataloaders['train_base'], device, epoch, lr, weight_decay, save_path, val_data_loader=dataloaders['test'], loss_save_path=loss_save_path)
    elif 'emb' in model_name:
        train_emb(model, dataloaders['train_base'], device, epoch, lr, weight_decay, save_path, val_data_loader=dataloaders['test'], loss_save_path=loss_save_path)
    else:
        train(model, dataloaders['train_base'], device, epoch, lr, weight_decay, save_path, val_data_loader=dataloaders['test'], loss_save_path=loss_save_path)
    
    torch.save(model.state_dict(), save_path)
    print("="*20, 'pretrain {}'.format(model_name), "="*20)
    

    loss, auc, f1 = test(model, dataloaders['train_warm_a'], device)
    print('pretrainloss {} pretrainauc {} pretrainf1 {}'.format(loss, auc, f1))
    return model, dataloaders, save_path


def emerg(model, dataloaders, model_name, epoch, meta_epoch, meta_lr, warm_lr, inner_lr, weight_decay, device, save_dir):
    save_path = os.path.join(save_dir, 'model.pth')
    warm_model = EmerG(model, dataloaders.item_features, device, inner_lr).to(device)
    criterion = torch.nn.BCELoss()
    warm_model.optimize_all()
    params_list = []
    for name, param in warm_model.named_parameters():
        if 'item_id' in name:
            torch.nn.init.uniform_(param, -0.01, 0.01)
        params_list.append(param)
    meta_optim = torch.optim.Adam(params=params_list, lr=meta_lr, weight_decay=weight_decay)
    def warm_up(dataset_spt, dataset_qry, epoch=0):
        for e in range(epoch):
            losses = []
            if dataset_qry:
                dataloader_qry_iter = iter(dataset_qry)
            for i, (features_spt, labels_spt) in enumerate(dataset_spt):
                features_qry, labels_qry = next(dataloader_qry_iter)
                features_spt = {k: v.view(-1, v.shape[-1]) for k, v in features_spt.items()}
                features_qry = {k: v.view(-1, v.shape[-1]) for k, v in features_qry.items()}
                warm_model.train()
                meta_optim.zero_grad()
                qry_labels_pred, loss_s = warm_model(features_spt, labels_spt, features_qry, inner_lr)
                
                loss_q = criterion(qry_labels_pred, labels_qry.view(-1, 1).float())
                loss_q = 0.1 * loss_s + 0.9 * loss_q
                losses.append(loss_q.item())
                loss_q.backward()
                meta_optim.step()
                warm_model.store_parameters()
                if i % 100 == 0:
                    print('epoch:{:2d}\titer:{:4d}\ttrain loss:{:.4f}'.format(e, i, np.mean(losses)), end='\r')
            loss, auc, f1 = test(warm_model.model, dataloaders['test'], device)
            print('[eval] epoch:{:2d}\tloss:{:.4f}\tauc:{:.4f}\tf1:{:.4f}'.format(e, loss, auc, f1))
            warm_model.model.global_graph = None
        return
    
    loss, auc, f1 = test(warm_model.model, dataloaders['test'], device)
    print('[init] loss:{:.4f}\tauc:{:.4f}\tf1:{:.4f}'.format(loss, auc, f1))
    warm_model.model.global_graph = None
    warm_model.model.warm = "generategraph"
    
    # meta training
    meta_dataloaders = [dataloaders[name] for name in ['metaE_a', 'metaE_b', 'metaE_c', 'metaE_d']]
    metagroup_dataloaders = [dataloaders[name] for name in ['metaa_item2group', 'metab_item2group', 'metac_item2group', 'metad_item2group']]
    auc_list, f1_list = [], []
    
    for i in range(meta_epoch):
        if i % 4 != 3:
            dataloader_a = metagroup_dataloaders[i % 4]
            dataloader_b = metagroup_dataloaders[(i + 1) % 4]
            warm_up(dataloader_a, dataloader_b, 1)
    
    

    
    # Warm Phase training and testing

    dataloader_warma_iter = iter(dataloaders['warma_item2group'])
    dataloader_warmb_iter = iter(dataloaders['warmb_item2group'])
    dataloader_warmc_iter = iter(dataloaders['warmc_item2group'])

    labels_cold, scores_cold, losses_cold = list(), list(), list()
    labels_a, scores_a, losses_a = list(), list(), list()
    labels_b, scores_b, losses_b = list(), list(), list()
    labels_c, scores_c, losses_c = list(), list(), list()
    
    for i, (features_test, label_test) in enumerate(dataloaders['test_item2group']):
        print("{}/{}".format(i, len(dataloaders['test_item2group'])), end='\r')
        warm_model_spe = deepcopy(warm_model)
        
        features_warma, label_warma = next(dataloader_warma_iter)
        features_warmb, label_warmb = next(dataloader_warmb_iter)
        features_warmc, label_warmc = next(dataloader_warmc_iter)
        while features_warma['item_id'].tolist()[0][0] != features_test['item_id'].tolist()[0][0]:
            features_warma, label_warma = next(dataloader_warma_iter)
            features_warmb, label_warmb = next(dataloader_warmb_iter)
            features_warmc, label_warmc = next(dataloader_warmc_iter)
        features_warma = {k: v.view(-1, v.shape[-1]) for k, v in features_warma.items()}
        features_warmb = {k: v.view(-1, v.shape[-1]) for k, v in features_warmb.items()}
        features_warmc = {k: v.view(-1, v.shape[-1]) for k, v in features_warmc.items()}
        features_test = {k: v.view(-1, v.shape[-1]) for k, v in features_test.items()}
        warm_model_spe.generate_graph(features_warma)
        
        warm_model_spe.model.eval()
        y = warm_model_spe.model(features_test)
        if True:
            label_test = label_test.to(device)
            if isinstance(label_test.tolist()[0], int):
                labels_cold.extend(label_test.tolist())
            else:
                labels_cold.extend(label_test.tolist()[0])
            scores_cold.extend(y.tolist())
            losses_cold.append(criterion(y.view(-1, 1), label_test.view(-1, 1).float()).item())

       
        warm_model_spe.model.optimize_itemid_graph()
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, warm_model_spe.model.parameters()), \
                                          lr=warm_lr, weight_decay=weight_decay)                      
        warm_model_spe.model.train()
        for epoch_i in range(epoch + 1):
            y = warm_model_spe.model(features_warma)
            loss = criterion(y.view(-1, 1), label_warma.view(-1, 1).float())
            warm_model_spe.model.zero_grad()
            loss.backward()
            optimizer.step()
        
        warm_model_spe.model.eval()
        y = warm_model_spe.model(features_test)
        if True:
            
            label_test = label_test.to(device)
            if isinstance(label_test.tolist()[0], int):
                labels_a.extend(label_test.tolist())
            else:
                labels_a.extend(label_test.tolist()[0])
            scores_a.extend(y.tolist())
            losses_a.append(criterion(y.view(-1, 1), label_test.view(-1, 1).float()).item())
        
        warm_model_spe.model.train()
         

        for epoch_i in range(epoch + 1):
            y = warm_model_spe.model(features_warmb)
            loss = criterion(y.view(-1, 1), label_warmb.view(-1, 1).float())
            warm_model_spe.model.zero_grad()
            loss.backward()
            optimizer.step()

        warm_model_spe.model.eval()
        y = warm_model_spe.model(features_test)
        if True:
            label_test = label_test.to(device)
            if isinstance(label_test.tolist()[0], int):
                labels_b.extend(label_test.tolist())
            else:
                labels_b.extend(label_test.tolist()[0])
            scores_b.extend(y.tolist())
            losses_b.append(criterion(y.view(-1, 1), label_test.view(-1, 1).float()).item())
        
        warm_model_spe.model.train()
         
        
        for epoch_i in range(epoch + 1):
            y = warm_model_spe.model(features_warmc)
            loss = criterion(y.view(-1, 1), label_warmc.view(-1, 1).float())
            warm_model_spe.model.zero_grad()
            loss.backward()
            optimizer.step()

        warm_model_spe.model.eval()
        y = warm_model_spe.model(features_test)
        if True:
            label_test = label_test.to(device)
            if isinstance(label_test.tolist()[0], int):
                labels_c.extend(label_test.tolist())
            else:
                labels_c.extend(label_test.tolist()[0])
            scores_c.extend(y.tolist())
            losses_c.append(criterion(y.view(-1, 1), label_test.view(-1, 1).float()).item())
        
    
    scores_arr_cold = np.array(scores_cold)
    scores_arr_a = np.array(scores_a)
    scores_arr_b = np.array(scores_b)
    scores_arr_c = np.array(scores_c)
    auc_list.append(roc_auc_score(labels_cold, scores_cold))
    auc_list.append(roc_auc_score(labels_a, scores_a))
    auc_list.append(roc_auc_score(labels_b, scores_b))
    auc_list.append(roc_auc_score(labels_c, scores_c))
    f1_list.append(f1_score(labels_cold, (scores_arr_cold > np.mean(scores_arr_cold)).astype(np.float32).tolist()))
    f1_list.append(f1_score(labels_a, (scores_arr_a > np.mean(scores_arr_a)).astype(np.float32).tolist()))
    f1_list.append(f1_score(labels_b, (scores_arr_b > np.mean(scores_arr_b)).astype(np.float32).tolist()))
    f1_list.append(f1_score(labels_c, (scores_arr_c > np.mean(scores_arr_c)).astype(np.float32).tolist()))
    print(auc_list)
    print(f1_list)
    return auc_list, f1_list




def run(model, dataloaders, args, model_name, warm):
    if warm == 'emerg':
        auc_list, f1_list = emerg(model, dataloaders, model_name, args.epoch, args.melu_epochs, args.meta_lr, args.warm_lr, args.lr_inner, args.weight_decay, args.device, args.save_dir)
    return auc_list, f1_list

if __name__ == '__main__':
    # init params
    #Hyper-parameters for movielens1M
    params = {'cold_phase_train_epochs': 2, 'pretrain_epochs': 2, 'batch_size': 512, 'lr': 0.001, 'lr_inner': 0.01, 'lr_pretrain': 0.005, 'meta_lr': 0.001, 'warm_lr': 0.01, 'epoch': 11, 'cvar_epochs': 2, 'melu_epochs': 11, 'is_dropoutnet': False, 'weight_decay': 2.9214406619851726e-07}
    #Hyper-parameters for taobaoAD
    params = {'cold_phase_train_epochs': 2, 'pretrain_epochs': 1, 'batch_size': 512, 'lr': 0.00195957510695886, 'lr_inner': 0.001, 'lr_pretrain': 0.001, 'meta_lr': 0.0001, 'warm_lr': 0.01, 'epoch': 16, 'cvar_epochs': 3, 'melu_epochs': 6, 'is_dropoutnet': False, 'weight_decay': 2.9214406619851726e-07}
#  get params from nni
    optimized_params = nni.get_next_parameter()
    print(optimized_params)
    params.update(optimized_params)
    print(params)

    args = get_args()
    args.bsz = params['batch_size']
    args.lr = params['lr']
    args.lr_inner = params['lr_inner']
    args.lr_pretrain = params['lr_pretrain']
    args.epoch = params['epoch']
    args.pretrain_epochs = params['pretrain_epochs']
    args.cvar_epochs = params['cvar_epochs']
    args.melu_epochs = params['melu_epochs']
    args.cold_phase_train_epochs = params['cold_phase_train_epochs']
    args.is_dropoutnet = params['is_dropoutnet']
    args.weight_decay = params['weight_decay']
    args.meta_lr = params['meta_lr']
    args.warm_lr = params['warm_lr']

    res = {}
    print(args.model_name)
    torch.cuda.empty_cache()

    avg_auc_list, avg_f1_list = [], []
    seeds = [7788, 1234, 9999] # seeds setting
    for i in range(args.runs):
        torch.cuda.empty_cache()
        seed = seeds[i]
        set_seed(seed)
        # load or train pretrain models
        model, dataloaders, save_path = pretrain(args.dataset_name, args.datahub_path, args.bsz, args.shuffle, args.model_name, \
            args.pretrain_epochs, args.lr_pretrain, args.weight_decay, seed, args.device, args.save_dir, args.is_dropoutnet)
            

        # warmup train and test
        model_v = get_model(args.model_name, dataloaders).to(args.device)
        if os.path.exists(save_path):
            model_v.load_state_dict(torch.load(save_path))
            print('model loaded!')

        auc_list, f1_list = run(model_v, dataloaders, args, args.model_name, args.warmup_model)
        avg_auc_list.append(np.array(auc_list))
        avg_f1_list.append(np.array(f1_list))
    
    avg_auc_list, std_auc_list = list(np.stack(avg_auc_list).mean(axis=0)), list(np.std(np.stack(avg_auc_list), axis=0))
    avg_f1_list, std_f1_list = list(np.stack(avg_f1_list).mean(axis=0)), list(np.std(np.stack(avg_f1_list), axis=0))
    print('     cold wm-a wm-b wm-c')
    print("auc: %.4f %.4f %.4f %.4f" % tuple(avg_auc_list))
    print("+- : %.4f %.4f %.4f %.4f" % tuple(std_auc_list))
    print("f1 : %.4f %.4f %.4f %.4f" % tuple(avg_f1_list))
    print("+- : %.4f %.4f %.4f %.4f" % tuple(std_f1_list))
