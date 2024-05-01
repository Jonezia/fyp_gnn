import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import scipy.io as sio
import networkx as nx
from collections import defaultdict
import torch.nn as nn
import torch
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from networkx.readwrite import json_graph
import json
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import sys
import os

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import NELL
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit2
from torch_geometric.transforms import NormalizeFeatures

def load_data_pyg(dataset_str, normalize=True):
    print(f"Loading {dataset_str} Dataset...")

    data = None
    if dataset_str == "cora":
        data = Planetoid(root='./data/Cora', name='Cora')
    elif dataset_str == "citeseer":
        data = Planetoid(root='./data/Citeseer', name='Citeseer')
    elif dataset_str == "pubmed":
        data = Planetoid(root='./data/Pubmed', name='Pubmed')
    elif dataset_str == "NELL":
        data = NELL(root='./data/NELL', transform=NormalizeFeatures())
    elif dataset_str == "flickr":
        data = Flickr(root='./data/Flickr', transform=NormalizeFeatures())
    elif dataset_str == "reddit2":
        data = Reddit2(root="./data/Reddit2", transform=NormalizeFeatures())
    else:
        raise ValueError("Not valid dataset")

    # Read Labels
    feat_data = data.x.numpy()
    # Read Features
    labels = data.y.numpy()
    # Read Edges
    edges = data.edge_index
    edges = edges.numpy().T

    if len(labels.shape) > 1:
        multilabel = True
        num_classes = len(labels[0])
    else:
        multilabel = False
        num_classes = np.max(labels) + 1

    train_nodes = np.array(np.squeeze(np.nonzero(data.train_mask)))
    val_nodes = np.array(np.squeeze(np.nonzero(data.val_mask)))
    test_nodes = np.array(np.squeeze(np.nonzero(data.test_mask)))

    print_statistics(edges, labels, feat_data, num_classes, train_nodes, val_nodes, test_nodes, multilabel)
            
    return edges, labels, feat_data, num_classes, train_nodes, val_nodes, test_nodes, multilabel

def print_statistics(edges, labels, feat_data, num_classes, train_nodes, val_nodes, test_nodes, multilabel):
    print("=============== Dataset Properties ===============")
    print(f"Total Nodes: {feat_data.shape[0]}")
    print(f"Total Edges: {edges.shape[0]}")
    print(f"Number of Features: {feat_data.shape[1]}")
    print(f"Number of Classes: {num_classes}")
    if not multilabel:
        print("Task Type: Multi-class Classification")
    else:
        print("Task Type: Multi-label Classification")
    print(f"Training Nodes: {len(train_nodes)}")
    print(f"Validation Nodes: {len(val_nodes)}")
    print(f"Testing Nodes: {len(test_nodes)}")
    print()

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str, normalize=True):
    """
    Loads a dataset

    Params:
        dataset_str (str): dataset name: Cora/CiteSeer/PubMed/PPI/Reddit
    Returns:
        edges: 
        labels: 
        feat_data: 
        num_classes: 
        idx_train: train node indices
        idx_val: val node indices
        idx_test: test node indices

        np.array(edges), np.array(degrees), np.array(labels), np.array(features),\
                np.array(idx_train), np.array(idx_val), np.array(idx_test)
        np.array(edges), labels, features, np.max(labels)+1,  idx_train, idx_val, idx_test
    """ 
    if dataset_str in ['ppi', 'toy-ppi', 'reddit']:
        prefix = f'data/{dataset_str}/{dataset_str}'
        G_data = json.load(open(prefix + "-G.json"))
        G = json_graph.node_link_graph(G_data)
        if isinstance(G.nodes()[0], int):
            conversion = lambda n : int(n)
        else:
            conversion = lambda n : n

        if os.path.exists(prefix + "-feats.npy"):
            feats = np.load(prefix + "-feats.npy")
        else:
            print("No features present.. Only identity features will be used.")
            feats = None
        id_map = json.load(open(prefix + "-id_map.json"))
        id_map = {conversion(k):int(v) for k,v in id_map.items()}
        class_map = json.load(open(prefix + "-class_map.json"))
        if isinstance(list(class_map.values())[0], list):
            lab_conversion = lambda n : n
            multilabel = True
        else:
            lab_conversion = lambda n : int(n)
            multilabel = False

        class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

        print(f"nodes: {len(G.nodes)}")
        print(f"edges: {len(G.edges)}")

        ## Remove all nodes that do not have val/test annotations
        ## (necessary because of networkx weirdness with the Reddit data)
        broken_count = 0
        to_remove = []
        for node in G.nodes():
            if not 'val' in G.nodes[node] or not 'test' in G.nodes[node]:
                to_remove.append(node)
                broken_count += 1
        for node in to_remove:
            G.remove_node(node)
        print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

        print(f"nodes: {len(G.nodes)}")
        print(f"edges: {len(G.edges)}")

        ## Make sure the graph has edge train_removed annotations
        ## (some datasets might already have this..)
        print("Loaded data.. now preprocessing..")
        for edge in G.edges():
            if (G.nodes[edge[0]]['val'] or G.nodes[edge[1]]['val'] or
                G.nodes[edge[0]]['test'] or G.nodes[edge[1]]['test']):
                G[edge[0]][edge[1]]['train_removed'] = True
            else:
                G[edge[0]][edge[1]]['train_removed'] = False

        if normalize and not feats is None:
            train_ids = np.array([id_map[str(n)] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
            train_feats = feats[train_ids]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            features = scaler.transform(feats)
        
        # degrees = np.zeros(len(G), dtype=np.int64)
        edges = []
        labels = []
        idx_train = []
        idx_val   = []
        idx_test  = []
        for s in G:
            if G.nodes[s]['test']:
                idx_test += [s]
            elif G.nodes[s]['val']:
                idx_val += [s]
            else:
                idx_train += [s]
            for t in G[s]:
                edges += [[s, t]]
            # degrees[s] = len(G[s])
            labels += [class_map[str(s)]]

        if multilabel:
            num_labels = len(labels[0])
        else:
            num_labels = np.max(labels) + 1

        edges = np.array(edges)
        labels = np.array(labels)
        feat_data = np.array(features)
        train_nodes = np.array(idx_train)
        val_nodes = np.array(idx_val)
        test_nodes = np.array(idx_test)

        print_statistics(edges, labels, feat_data, num_labels, train_nodes, val_nodes, test_nodes, multilabel)
        
        return edges, labels, feat_data, num_labels, train_nodes, val_nodes, test_nodes, multilabel
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(f"data/{dataset_str}/ind.{dataset_str}.{names[i]}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"data/{dataset_str}/ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = np.array(test_idx_range.tolist())
    idx_train = np.array(range(len(y)))
    idx_val = np.array(range(len(y), len(y)+500))

    degrees = np.zeros(len(labels), dtype=np.int64)
    edges = []
    for s in graph:
        for t in graph[s]:
            edges += [[s, t]]
        degrees[s] = len(graph[s])
    labels = np.argmax(labels, axis=1)

    print_statistics(np.array(edges), labels, features, np.max(labels)+1, idx_train, idx_val, idx_test, False)

    return np.array(edges), labels, features, np.max(labels)+1,  idx_train, idx_val, idx_test, False

def sym_normalize(mx):
    """Sym-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    
    colsum = np.array(mx.sum(0))
    c_inv = np.power(colsum, -1/2).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)
    
    mx = r_mat_inv.dot(mx).dot(c_mat_inv)
    return mx

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx

def get_sparse(edges, num_nodes):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj) 

def norm(l):
    return (l - np.average(l)) / np.std(l)

def stat(l):
    return np.average(l), np.sqrt(np.var(l))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices, values, shape


def get_adj(edges, num_nodes):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes), dtype=np.float32)
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

def get_laplacian(adj):
    adj = row_normalize(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj)

def logit_to_label(out):
    return out.argmax(dim=1)

def metrics(logits, y):
    if isinstance(y, torch.LongTensor) and y.dim() == 1 or y.ndim == 1: # Multi-class
        y_pred = logit_to_label(logits)

        if isinstance(y, torch.LongTensor):
            y = y.cpu()
        
        cm = confusion_matrix(y, y_pred.cpu())
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
    
        acc = np.diag(cm).sum() / cm.sum()
        micro_f1 = acc # micro f1 = accuracy for multi-class
        sens = TP.sum() / (TP.sum() + FN.sum())
        spec = TN.sum() / (TN.sum() + FP.sum())
    
    else: # Multi-label
        y_pred = logits >= 0.5
        
        tp = int((y & y_pred).sum())
        tn = int((~y & ~y_pred).sum())
        fp = int((~y & y_pred).sum())
        fn = int((y & ~y_pred).sum())
        
        acc = (tp + tn)/(tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        micro_f1 = 2 * (precision * recall) / (precision + recall)
        sens = (tp)/(tp + fn)
        spec = (tn)/(tn + fp)
        
    return acc, micro_f1, sens, spec

from scipy.sparse.csgraph import shortest_path
# uses Dijkstra's algorithm, O[N(N*k + N*log(N))],
# where k is the average number of connected edges per node and N is |V|
def find_overlapping_k_hops(adj_matrix, indices, k):
    # Compute the shortest path distances from the starting nodes
    distances = shortest_path(adj_matrix, indices=indices, return_predecessors=False, method="D")
    # Find the indices of all nodes within k hops
    within_k_hops = np.nonzero(np.any(distances <= k, axis=0))[0]
    return within_k_hops

from collections import deque
# Uses BFS, O(|V| + |E|) i.e. good for sparse matrices
def find_overlapping_k_hops_bfs(adj_matrix, indices, k):
    n = adj_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    queue = deque(indices)
    within_k_hops = set(indices)

    for _ in range(k):
        for _ in range(len(queue)):
            node = queue.popleft()
            for neighbor in adj_matrix.getrow(node).indices:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    within_k_hops.add(neighbor)

    return np.array(list(within_k_hops))

def roundsize(size):
    # returns size in MB
    return round(size / 1024 / 1024,4)

def mean_and_std(array, decimals=2):
    # returns mean & std dev of numpy array as string
    return f"{round(np.average(array),decimals)}±{round(np.std(array),decimals)}"

def print_report(args, log_times, log_total_iters, log_best_epoch, log_max_memory,
                 log_test_acc, log_test_f1, log_test_sens, log_test_spec):
    print()
    print(f"==== {args.dataset} {args.sampler} {args.model} {args.n_layers}layer results ====")
    print(f"Time:           {mean_and_std(log_times)}")
    print(f"Epochs:         {mean_and_std(log_total_iters)}")
    print(f"Best epoch:     {mean_and_std(log_best_epoch)}")
    print(f"Time per epoch: {mean_and_std(np.array(log_times) / np.array(log_total_iters), 3)}")
    print(f"Max memory:     {mean_and_std(log_max_memory)}")
    print(f"Accuracy:       {mean_and_std(log_test_acc, 3)}")
    print(f"F1:             {mean_and_std(log_test_f1, 3)}")
    print(f"Sensitivity:    {mean_and_std(log_test_sens, 3)}")
    print(f"Specificity:    {mean_and_std(log_test_spec, 3)}")
