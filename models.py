import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=bias)
    def forward(self, x, adj):
        out = self.linear(x)
        return F.elu(torch.spmm(adj, out))

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout, nout):
        super(GCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid,  nhid))
        self.linear =  nn.Linear(nhid, nout)
    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if type(adjs) is list:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[idx]))
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs))
        return self.linear(x)

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
    def __init__(self, nfeat, nhid, layers, dropout, nout):
        super(ScalarGCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(ScalarGraphConvolution())
        self.linear =  nn.Linear(nhid, nout)
    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if type(adjs) is list:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[idx]))
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs))
        return self.linear(x)
    
# Same as ScalarGCN but with fixed constant of 1 instead of learnable param
class FixedScalarGraphConvolution(nn.Module):
    def __init__(self):
        super(FixedScalarGraphConvolution, self).__init__()
    def forward(self, x, adj):
        return F.elu(torch.spmm(adj, x))

class FixedScalarGCN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout, nout):
        super(FixedScalarGCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(FixedScalarGraphConvolution())
        self.linear =  nn.Linear(nhid, nout)
    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if type(adjs) is list:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[idx]))
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs))
        return self.linear(x)

# Same as ScalarGCN but without up-down feature transformation
class ScalarGCNNoFeatureTrans(nn.Module):
    def __init__(self, nfeat, layers, dropout, nout):
        super(ScalarGCNNoFeatureTrans, self).__init__()
        self.layers = layers
        self.gcs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(layers):
            self.gcs.append(ScalarGraphConvolution())
        self.linear =  nn.Linear(nfeat, nout)
    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if type(adjs) is list:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[idx]))
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs))
        return self.linear(x)
    
class SGC(nn.Module):
    def __init__(self, nfeat, layers, dropout, nout):
        super(SGC, self).__init__()
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.linear =  nn.Linear(nfeat, nout)
    def forward(self, x, adj_k):
        x = self.dropout(torch.spmm(adj_k, x))
        return self.linear(x)
    
class SIGNConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(SIGNConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=bias)
    def forward(self, x, adj):
        out = self.linear(x)
        return torch.spmm(adj, out)

class SIGN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout, nout):
        super(SIGN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(layers + 1):
            self.gcs.append(SIGNConvolution(nfeat, nhid))
        self.linear =  nn.Linear(nhid * (layers + 1), nout)
    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        concat = torch.tensor([]).to(torch.device("cuda:0"))
        for idx in range(len(self.gcs)):
            out = self.dropout(self.gcs[idx](x, adjs[idx]))
            concat = torch.cat((concat, out), dim=1)
        x = self.dropout(F.elu(concat))
        return self.linear(x)

# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat

#         self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         self.a = nn.Parameter(torch.FloatTensor(2*out_features, 1))
#         self.init_weights()

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def init_weights(self):
#         nn.init.xavier_uniform_(self.W.data)
#         nn.init.xavier_uniform_(self.a.data)

#     def forward(self, h, adj):
#         print(f"h.shape {h.shape}")

#         Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features=D)
#         N = Wh.size(0)
#         print(f"Wh.shape {Wh.shape}")

#         # Wh.repeat(1, N).view(N * N, -1) -> (N, N * D) -> (N * N, D)
#         # Wh.repeat(N, 1) -> (N * N, D)
#         # torch.cat(...) -> (N * N, 2D)
#         # .view(...) -> (N, N, 2D)
#         a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

#         print(f"adj.shape {adj.shape}")
#         print(f"e.shape {e.shape}")

#         # TODO: Adj input should be sparse, Attention should be sparse
#         # attention = torch.where(adj > 0, e, zero_vec)
#         crow_indices = adj.crow_indices()
#         print(crow_indices)
#         col_indices = adj.col_indices()
#         print(col_indices)
#         print(adj.values())
#         # Initialize a new tensor for the replaced values
#         new_values = torch.tensor([])

#         # Replace the values with the corresponding values from the dense matrix
#         for row in range(len(crow_indices) - 1):
#             start_idx = crow_indices[row]
#             end_idx = crow_indices[row + 1]
#             cols = col_indices[start_idx:end_idx]
#             new_values = torch.cat((new_values, e[row, cols]), dim=0)
        
#         attention = torch.sparse.csr_tensor(adj.crow_indices(), adj.col_indices(), new_values, size=adj.shape, dtype=torch.float32)
#         attention = torch.sparse.softmax(attention, dim=1)
#         # attention = F.dropout(attention, self.dropout)
#         h_prime = torch.spmm(attention, Wh)

#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime

# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads):
#         super(GAT, self).__init__()
#         self.dropout = dropout
#         self.layers = layers

#         # First GAT layer
#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         # Hidden GAT layers
#         for layer in range(1, layers):
#             setattr(self, f'attlayer_{layer}', [GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
#             for i, attention in enumerate(getattr(self, f'attlayer_{layer}')):
#                 self.add_module(f'attention_{layer}_{i}', attention)

#         # Output GAT layer
#         self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout)

#         # First GAT layer
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout)

#         # Hidden GAT layers
#         for layer in range(1, self.layers):
#             attlayer = getattr(self, f'attlayer_{layer}')
#             x = torch.cat([att(x, adj) for att in attlayer], dim=1)
#             x = F.dropout(x, self.dropout)

#         # Output GAT layer
#         x = self.out_att(x, adj)
#         return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2*out_features, 1))
        self.init_weights()

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def init_weights(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, h, adj):

        print(f"adj.shape {adj.shape}")

        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features=D)
        N = Wh.size(0)

        print(f"Wh.shape {Wh.shape}")

        # Wh.repeat(1, N).view(N * N, -1) -> (N, N * D) -> (N * N, D)
        # Wh.repeat(N, 1) -> (N * N, D)
        # torch.cat(...) -> (N * N, 2D)
        # .view(...) -> (N, N, 2D)
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout)
        h_prime = torch.mm(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads):
#         super(GAT, self).__init__()
#         self.dropout = dropout
#         self.layers = layers

#         # First GAT layer
#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         # Hidden GAT layers
#         for layer in range(1, layers):
#             setattr(self, f'attlayer_{layer}', [GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
#             for i, attention in enumerate(getattr(self, f'attlayer_{layer}')):
#                 self.add_module(f'attention_{layer}_{i}', attention)

#         # Output GAT layer
#         self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

#     def forward(self, x, adj):
#         adj = adj.to_dense()

#         x = F.dropout(x, self.dropout)

#         # First GAT layer
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout)

#         # Hidden GAT layers
#         for layer in range(1, self.layers):
#             attlayer = getattr(self, f'attlayer_{layer}')
#             x = torch.cat([att(x, adj) for att in attlayer], dim=1)
#             x = F.dropout(x, self.dropout)

#         # Output GAT layer
#         x = self.out_att(x, adj)
#         return x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layers = layers
        self.gcs = nn.ModuleList()
        # Initial layer
        self.gcs.append(nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]))
        self.dropout = nn.Dropout(dropout)
        # Hidden layers
        for i in range(layers-1):
            self.gcs.append(nn.ModuleList([GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]))
        self.linear =  nn.Linear(nhid, nout)
        # Output GAT layer
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adjs):
        if type(adjs) is list:
            for idx in range(len(self.gcs)):
                adj = adjs[idx].to_dense()
                x = self.dropout(torch.cat([att(x, adj) for att in self.gcs[idx]], dim=1))
            x = self.out_att(x, adjs[-1])
        else:
            adjs = adjs.to_dense()
            for idx in range(len(self.gcs)):
                x = self.dropout(torch.cat([att(x, adjs) for att in self.gcs[idx]], dim=1))
            x = self.out_att(x, adjs)
        return x