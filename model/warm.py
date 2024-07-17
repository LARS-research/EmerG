import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import os
import re
import pickle as pkl
from copy import deepcopy
from collections import OrderedDict
import time

class DropoutNet(nn.Module):

    def __init__(self, model: nn.Module, device, item_id_name='item_id'):
        super(DropoutNet, self).__init__()
        self.model = model
        self.item_id_name = item_id_name
        item_emb = self.model.emb_layer[self.item_id_name]
        self.mean_item_emb = torch.mean(item_emb.weight.data, dim=0, keepdims=True) \
                            .repeat(item_emb.num_embeddings, 1)
        return

    def foward_without_itemid(self, xdict):
        bsz = xdict[self.item_id_name].shape[0]
        target = self.model.forward_with_item_id_emb(self.mean_item_emb.repeat([bsz, 1]), xdict)
        return target

    def foward(self, xdict):
        item_id_emb = xdict[self.item_id_name]
        target = self.model.forward_with_item_id_emb(item_id_emb, xdict)
        return target
    

class EmerG(torch.nn.Module):
    def __init__(self, model: nn.Module, item_features, device, local_lr):
        super(EmerG, self).__init__()
        self.item_features = item_features
        self.model = model
        self.device = device
        self.local_lr = local_lr
        self.store_parameters()
        self.criterion = torch.nn.BCELoss()
        self.graph_gen = GenerateGraph(len(self.item_features), self.model.num_fields, self.model.embed_dim, device, self.model.gnn_layers)

    def store_parameters(self):
        self.keep_model = deepcopy(self.model)
        self.keep_weight = self.keep_model.state_dict()
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)

    @torch.no_grad()
    def reset_model(self):
        for p1, p2 in zip(self.model.parameters(), self.keep_model.parameters()):
            p1.data = p2.data

    def generate_graph(self, x_dict):
        embs = []
        embs.append(self.model.emb_layer[self.model.item_id_name](x_dict[self.model.item_id_name]).squeeze(1))
        for name, (_, type) in self.model.description.items():
            if name in self.item_features:
                if type == 'label' or name == self.model.item_id_name:
                    continue
                x = x_dict[name]
                if type == 'spr' and name != 'zip-code':
                    embs.append(self.model.emb_layer[name](x).squeeze(1))
                elif type == 'spr' and name == 'zip-code':
                    embs.append(self.model.emb_layer['zipcode'](x).squeeze(1))
                elif type == 'ctn' and name != 'count' and name != 'time_stamp':
                    embs.append(self.model.ctn_emb_layer[name] * x)
                elif type == 'seq':
                    embs.append(self.model.emb_layer[name](x).sum(dim=1))
                else:
                    pass
        emb_graph = torch.cat([emb for emb in embs], dim=1)
        graphs = self.graph_gen(emb_graph)
        for i in range(self.model.gnn_layers):
            origin_graph_dict = self.model.graph_dict.weight.data
            indexes = x_dict[self.model.item_id_name].squeeze()
            origin_graph_dict[indexes, ] = graphs.reshape(graphs.shape[0], self.model.num_fields * self.model.num_fields)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def optimize_gen(self):
        for name, param in self.named_parameters():
            if 'graph_gen' in name or 'graph_dicts' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def forward(self, x_dict_support, label_support, x_dict_query, inner_lr):
        
        self.generate_graph(x_dict_support)
        
        pred_support = self.model(x_dict_support)
        loss_support = self.criterion(pred_support, label_support.view(-1, 1).float())
        self.model.zero_grad()
        grad = torch.autograd.grad(loss_support, self.model.parameters(), create_graph=True, allow_unused=True)
        with torch.no_grad():
            for i in range(self.weight_len):
                if 'graph_dict' in self.weight_name[i] or 'item_id' in self.weight_name[i]:
                    match = re.search('(\.[0-9]+)\.', self.weight_name[i])
                    if match is None:
                        weight_name = 'self.model.' + self.weight_name[i]
                    else:
                        num = match.group(1)
                        weight_name = 'self.model.' + self.weight_name[i].replace(num, f'[{num[1:]}]')
                    eval(f'{weight_name}.set_({weight_name}.data - inner_lr * grad[i])')
        
        pred_query = self.model(x_dict_query)
        self.reset_model()
        return pred_query, loss_support
        
    def forward_for_metalearning(self, x_dict_support, label_support, x_dict_query, inner_lr):
        
        self.generate_graph(x_dict_support)
        
        pred_support = self.model(x_dict_support)
        loss_support = self.criterion(pred_support, label_support.view(-1, 1).float())
        
        pred_query = self.model(x_dict_query)
        self.reset_model()
        return pred_query, loss_support
    


class GenerateGraph(nn.Module):
    def __init__(self, num_item_fields, num_fields, embedding_dim, device=None, gnn_layers=0):
        super(GenerateGraph, self).__init__()
        self.device = device
        self.gnn_layers = gnn_layers
        self.graphgenerator = GraphGenerator(num_item_fields, num_fields, embedding_dim, device)

    def forward(self, feature_emb):
        graph = self.graphgenerator(feature_emb)
        return graph


class GraphGenerator(nn.Module):
    def __init__(self, num_item_fields, num_fields, embedding_dim, device):
        super(GraphGenerator, self).__init__()
        self.num_fields = num_fields
        self.num_item_fields = num_item_fields
        self.device = device
        self.generators = nn.ModuleList([nn.Sequential(
            nn.Linear(num_item_fields * embedding_dim + num_fields, num_fields),
            nn.ReLU(),
            nn.Linear(num_fields, num_fields),
            nn.ReLU(),
            nn.Linear(num_fields, num_fields)
        ) for _ in range(num_fields)])


    def forward(self, feature_emb):
        #print("begin gen graph")
        #begin_time = time.time()
        graph_fields = []
        for i in range(self.num_fields):
            field_index = torch.tensor([i]).to(self.device)
            field_onehot = F.one_hot(field_index, num_classes=self.num_fields).repeat(feature_emb.shape[0], 1)
            graph_field = self.generators[i](torch.cat([feature_emb, field_onehot], dim=1))
            graph_fields.append(graph_field)
        graph = torch.cat([graph_field.unsqueeze(1) for graph_field in graph_fields], dim=1)
        task_size = graph.shape[0]
        graph = torch.mean(graph, dim=0).squeeze()
        graph = graph.unsqueeze(0).repeat(task_size, 1, 1)
        #print("end gen graph")
        #print(time.time() - begin_time)
        return graph




        



