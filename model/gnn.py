from ast import iter_fields
import torch
import numpy as np
from copy import deepcopy
from itertools import product
from typing import Optional
from torch.autograd import Variable
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from collections import OrderedDict
import csv
import codecs
import time

class GNN(torch.nn.Module):
    def __init__(self, 
                 description,
                 embed_dim,
                 gnn_layers=3,
                 use_residual=True,
                 use_gru=False,
                 reuse_graph_layer=False,
                 device=None,
                 item_id_name='item_id',
                 one_w=True,
                 have_graph=True):
        super(GNN, self).__init__()
        self.features = [name for name, _, type in description if type != 'label']
        assert item_id_name in self.features, 'unkown item id name'
        self.description = {name: (size, type) for name, size, type in description}
        self.num_fields = len([type for _, _, type in description if type in ['spr', 'ctn', 'seq']]) - 2
        self.item_id_name = item_id_name
        self.device = device
        self.one_w = one_w
        self.global_graph = None
        self.have_graph = have_graph
        self.gnn_layers = gnn_layers
        self.embed_dim = embed_dim
        self.warm = "basegraph"
        self.criterion = torch.nn.BCELoss()
        self.build(embed_dim, gnn_layers, reuse_graph_layer, use_gru, use_residual, device)
    
    def build(self, embed_dim, gnn_layers, reuse_graph_layer, use_gru, use_residual, device):
        self.emb_layer = torch.nn.ModuleDict()
        self.ctn_emb_layer = torch.nn.ParameterDict()
        self.embed_output_dim = 0
        for name, (size, type) in self.description.items():
            if type == 'spr' and name != 'zip-code':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                self.embed_output_dim += embed_dim
            elif type == 'spr' and name == 'zip-code':
                self.emb_layer['zipcode'] = torch.nn.Embedding(size, embed_dim)
                self.embed_output_dim += embed_dim
            elif type == 'ctn' and name != 'count' and name != 'time_stamp':
                self.ctn_emb_layer[name] = torch.nn.Parameter(torch.zeros([1, embed_dim], requires_grad=True))
                self.embed_output_dim += embed_dim
            elif type == 'seq':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                self.embed_output_dim += embed_dim
            elif type == 'label':
                pass
            else:
                pass
            
            if name == self.item_id_name:
                self.graph_dict = torch.nn.Embedding(size, self.num_fields * self.num_fields)

        if self.have_graph:
            self.gnn = GNN_Layer(self.num_fields, 
                                    embed_dim,
                                    gnn_layers=gnn_layers,
                                    reuse_graph_layer=reuse_graph_layer,
                                    use_gru=use_gru,
                                    use_residual=use_residual,
                                    device=device,
                                    one_w=self.one_w)
        self.fc = AttentionalPrediction(self.num_fields, embed_dim)
    
    def init(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def optimize_itemid_graph(self):
        for name, param in self.named_parameters():
            if self.item_id_name not in name and 'graph_dict' not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        return

    def forward(self, x_dict):
        embs = []
        embs.append(self.emb_layer[self.item_id_name](x_dict[self.item_id_name]).squeeze(1))
        for name, (_, type) in self.description.items():
            if type == 'label' or name == self.item_id_name:
                continue
            x = x_dict[name]
            if type == 'spr' and name != 'zip-code':
                embs.append(self.emb_layer[name](x).squeeze(1))
            elif type == 'spr' and name == 'zip-code':
                embs.append(self.emb_layer['zipcode'](x).squeeze(1))
            elif type == 'ctn' and name != 'count' and name != 'time_stamp':
                embs.append(self.ctn_emb_layer[name] * x)
            elif type == 'seq':
                embs.append(self.emb_layer[name](x).sum(dim=1))
            else:
                pass

        emb = torch.cat([emb.unsqueeze(1) for emb in embs], dim=1)
        graphs = []
        if len(graphs) == 0:
            if self.warm == "generategraph":
                #print("begin iter graph")
                #begin_time = time.time()
                x = x_dict[self.item_id_name]
                graph = self.graph_dict(x)
                
                task_size = graph.shape[0]
                graph = graph.reshape(graph.shape[0], self.num_fields, self.num_fields)
                '''
                graph = torch.mean(graph, dim=0).squeeze()
                
                min_graph = torch.min(graph)
                max_graph = torch.max(graph)
                graph = (graph - min_graph) / (max_graph - min_graph)
                mask = torch.eye(self.num_fields).to(self.device)
                graph = graph.masked_fill(mask.bool(), 1)

                graph = graph.reshape(self.num_fields * self.num_fields)
                path_choose = torch.ones(self.num_fields *self.num_fields).to(self.device)
                _, idx = graph.topk((self.num_fields * self.num_fields) // 2)
                path_choose[idx] = 0
                graph = graph.masked_fill(path_choose.bool(), 0)
                
                graph = graph.reshape(self.num_fields, self.num_fields)
                graph = graph.unsqueeze(0).repeat(task_size, 1, 1)


                graph_undirected = (graph + graph.transpose(1, 2)) / 2
                graph_undirected_bool = graph_undirected.bool()
                '''
                #graph_undirected_bool_clone = graph_undirected_bool.clone().detach()
                #print(graph_undirected_bool_clone)
                #graph_undirected_bool_clone[:, [self.num_fields - 2, self.num_fields - 1], :] = False
                #graph_undirected_bool_clone[:, :, [self.num_fields - 2, self.num_fields - 1]] = False
                #print(graph_undirected_bool_clone)

                for i in range(self.gnn_layers):   
                    '''
                    if i == 0:
                        graph = graph_undirected
                        graph_last = graph
                    else:
                        graph = torch.bmm(graph_last, graph_undirected)
                        min_graph = torch.min(graph)
                        max_graph = torch.max(graph)
                        graph = (graph - min_graph) / (max_graph - min_graph)
                        graph = graph * graph_undirected_bool
                        graph_last = graph
                    '''
                    graphs.append(graph)
                
                #print("end iter graph")
                #print(time.time() - begin_time)
                
            elif self.warm == "basegraph":
                graphs = None
            else:
                raise ValueError('False graphtype!')
        if self.have_graph:
            emb = self.gnn(graphs, emb, self.warm)
        y_pred = F.relu(emb)
        y_pred = self.fc(y_pred)
        return torch.sigmoid(y_pred)

class AttentionalPrediction(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super(AttentionalPrediction, self).__init__()
        self.mlp1 = nn.Linear(embedding_dim, 1, False)
        self.mlp2 = nn.Sequential(nn.Linear(num_fields * embedding_dim, num_fields, False),
                                  nn.Sigmoid())

    def forward(self, h):
        score = self.mlp1(h).squeeze(-1) # b x f
        weight = self.mlp2(h.flatten(start_dim=1)) # b x f
        logit = (weight * score).sum(dim=1).unsqueeze(-1)
        return logit


class GNN_Layer(nn.Module):
    graphs = None
    def __init__(self, 
                 num_fields, 
                 embedding_dim,
                 gnn_layers=3,
                 reuse_graph_layer=False,
                 use_gru=False,
                 use_residual=True,
                 device=None,
                 one_w=True
                 ):
        super(GNN_Layer, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers
        self.use_residual = use_residual
        self.reuse_graph_layer = reuse_graph_layer
        self.device = device
        self.print_graph = 0
        self.num_heads = 3
        self.dropout = 0.0
        self.use_scale = True
        self.use_layer_norm = False
        self.num_att_layers = 1

        if reuse_graph_layer:
            self.gnn = GraphLayer(num_fields, embedding_dim, one_w)
        else:
            self.gnn = nn.ModuleList([GraphLayer(num_fields, embedding_dim, one_w)
                                      for _ in range(gnn_layers)])
        self.gru = nn.GRUCell(embedding_dim, embedding_dim) if use_gru else None
        self.src_nodes, self.dst_nodes = zip(*list(product(range(num_fields), repeat=2)))
        self.relu = nn.LeakyReLU()
        self.basegraph = nn.ModuleList([nn.Linear(num_fields, num_fields, False) for _ in range(gnn_layers)])
        self.multi_attention = nn.Sequential(
            *[MultiHeadSelfAttention(self.embedding_dim,
                                    attention_dim=self.embedding_dim,
                                    num_fields = self.num_fields, 
                                    num_heads=self.num_heads, 
                                    dropout_rate=self.dropout,  
                                    use_scale=self.use_scale, 
                                    layer_norm=self.use_layer_norm,
                                    layer=i,
                                    num_att_layers=self.num_att_layers) 
            for i in range(self.num_att_layers)])

    def forward(self, graphs, feature_emb, warm):
        H = []
        if warm == "generategraph":
            g = graphs
            self.print_graph = self.print_graph + 1
            h = feature_emb
            final_h = feature_emb
            H.append(feature_emb)
            for i in range(self.gnn_layers):
                if self.reuse_graph_layer:
                    a = self.gnn(g[i], feature_emb)
                else:
                    a = self.gnn[i](g[i], feature_emb)
                if i != self.gnn_layers - 1:
                    a = self.relu(a)

                h = a * h
                final_h = final_h + h
                H.append(h)
        elif warm == "basegraph":
            h = feature_emb
            final_h = feature_emb
            H.append(feature_emb)
            for i in range(self.gnn_layers):
                a = self.basegraph[i](feature_emb.transpose(1, 2)).transpose(1, 2)
                if i != self.gnn_layers - 1:
                    a = self.relu(a)

                h = a * h
                final_h = final_h + h
                H.append(h)
        else:
            print("wrong graph type!!!")

        Hs = torch.cat([h.reshape(-1, self.num_fields * self.embedding_dim).unsqueeze(1) for h in H], dim=1)
        final_h = self.multi_attention(Hs)
        final_h = final_h.sum(dim=1)
        final_h = final_h.reshape(-1, self.num_fields, self.embedding_dim)

        return final_h


class GraphLayer(nn.Module):
    def __init__(self, num_fields, embedding_dim, one_w):
        super(GraphLayer, self).__init__()
        self.num_fields = num_fields
        self.one_w = one_w
        self.mask_test = torch.ones(num_fields, num_fields, device='cuda:0')
        self.mask_test[[self.num_fields - 4, self.num_fields - 3, self.num_fields - 2, self.num_fields - 1], :] = 0
        self.mask_test[:, [self.num_fields - 4, self.num_fields - 3, self.num_fields - 2, self.num_fields - 1]] = 0
        if self.one_w:
            self.W = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        

    def forward(self, g, h):
        g = g * self.mask_test
        if self.one_w:
            a = torch.matmul(h, self.W).squeeze(-1)
            a = torch.bmm(g, a)
        return a

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim=None, num_fields=None, num_heads=1, dropout_rate=0., 
                 use_scale=False, layer_norm=False, layer=None, num_att_layers=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.layer = layer
        self.num_att_layers = num_att_layers    
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.num_fields = num_fields
        self.scale = attention_dim ** 0.5 if use_scale else None
        if layer == 0:
            self.W_q = nn.Linear(self.embedding_dim * self.num_fields, self.attention_dim * self.num_fields * self.num_heads, bias=False)
        else:
            self.W_q = nn.Linear(self.attention_dim * self.num_fields * self.num_heads, self.attention_dim * self.num_fields * self.num_heads, bias=False)
        if layer == 0:
            self.W_k = nn.Linear(self.embedding_dim * self.num_fields, self.attention_dim * self.num_fields * self.num_heads, bias=False)
        else:
            self.W_k = nn.Linear(self.attention_dim * self.num_fields * self.num_heads, self.attention_dim * self.num_fields * self.num_heads, bias=False)
        if layer == 0:
            self.W_v = nn.Linear(self.embedding_dim, self.attention_dim * self.num_heads, bias=False)
        else:
            self.W_v = nn.Linear(self.attention_dim * self.num_heads, self.attention_dim * self.num_heads, bias=False)
        
        if self.layer == self.num_att_layers - 1:
            self.W_res = nn.Linear(self.attention_dim * self.num_heads, self.embedding_dim)
        else:
            self.W_res = None    
        
        self.dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, Hs, mask=None):
        H = Hs.reshape(Hs.size(0), Hs.size(1), self.num_fields, -1)

        query = self.W_q(Hs)
        key = self.W_k(Hs)
        value = self.W_v(H)
        value = value.reshape(Hs.size(0), Hs.size(1), -1)

        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.attention_dim)
        key = key.view(batch_size * self.num_heads, -1, self.attention_dim)
        value = value.view(batch_size * self.num_heads, -1, self.attention_dim)
        if mask:
            mask = mask.repeat(self.num_heads, 1, 1)
        output, attention = self.dot_product_attention(query, key, value, self.scale, mask)
        output = output.view(batch_size, -1, self.num_heads * self.attention_dim * self.num_fields)
        if self.W_res is not None:
            output = output.view(Hs.size(0), Hs.size(1), self.num_fields, -1) 
            output = self.W_res(output)
            output = output.view(Hs.size(0), Hs.size(1), -1)
        if self.dropout is not None:
            output = self.dropout(output)
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        
        return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, W_q, W_k, W_v, scale=None, mask=None):
        attention = torch.bmm(W_q, W_k.transpose(1, 2))
        if scale:
            attention = attention / scale
        if mask:
            attention = attention.masked_fill_(mask, -np.inf)
        attention = self.softmax(attention)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention, W_v)
        return output, attention

    


