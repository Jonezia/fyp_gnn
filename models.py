import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in,  n_out)
    def forward(self, x, adj):
        out = self.linear(x)
        return F.elu(torch.spmm(adj, out))

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout):
        super(GCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid,  nhid))
    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if len(adjs) > 1:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[idx]))
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[0]))
        return x

class SuGCN(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super(SuGCN, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(self.encoder.nhid, num_classes)
    def forward(self, feat, adjs):
        x = self.encoder(feat, adjs)
        x = self.dropout(x)
        x = self.linear(x)
        return x
    
class ScalarGraphConvolution(nn.Module):
    def __init__(self):
        super(ScalarGraphConvolution, self).__init__()
        self.scalar = nn.Parameter(torch.ones(1))
    def forward(self, x, adj):
        out = self.scalar * x
        return F.elu(torch.spmm(adj, out))

class ScalarGCN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout):
        super(ScalarGCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(ScalarGraphConvolution())
    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if len(adjs) > 1:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[idx]))
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[0]))
        return x

# scalar SGC is the same as ScalarGCN but without up-down feature transformation
class ScalarSGC(nn.Module):
    def __init__(self, nfeat, layers, dropout):
        super(ScalarSGC, self).__init__()
        self.layers = layers
        self.nhid = nfeat
        self.gcs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(layers):
            self.gcs.append(ScalarGraphConvolution())
    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if len(adjs) > 1:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[idx]))
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[0]))
        return x
    
# class SimpleGraphConvolution(nn.Module):
#     def __init__(self):
#         super(SimpleGraphConvolution, self).__init__()
#     def forward(self, x, adj):
#         return F.elu(torch.spmm(adj, x))
    
# class SGC(nn.Module):
#     def __init__(self, nfeat, layers, dropout):
#         super(SGC, self).__init__()
#         self.layers = layers
#         self.nhid = nfeat
#         self.gcs = nn.ModuleList()
#         self.dropout = nn.Dropout(dropout)
#         for i in range(layers):
#             self.gcs.append(SimpleGraphConvolution())
#     def forward(self, x, adjs):
#         '''
#             The difference here with the original GCN implementation is that
#             we will receive different adjacency matrix for different layer.
#         '''
#         for idx in range(len(self.gcs)):
#             x = self.dropout(self.gcs[idx](x, adjs[idx]))
#         return x