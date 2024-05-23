import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from utils import *

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
    def forward(self, x, adjs, sampled = None):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if sampled is not None:
            x = x[sampled[0]]
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[idx]))
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs))
        return self.linear(x)
    
class ScalarGraphConvolution(nn.Module):
    def __init__(self):
        super(ScalarGraphConvolution, self).__init__()
        self.scalar = nn.Parameter(torch.ones(1))
    def forward(self, x, adj):
        out = torch.mul(x, self.scalar)
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
    def forward(self, x, adjs, sampled = None):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if sampled is not None:
            x = x[sampled[0]]
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[idx]))
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs))
        return self.linear(x)
    
# Same as ScalarGCN but with a single shared scalar
class SingleScalarGraphConvolution(nn.Module):
    def __init__(self):
        super(SingleScalarGraphConvolution, self).__init__()
    def forward(self, x, adj, scalar):
        out = scalar * x
        return F.elu(torch.spmm(adj, out))

class SingleScalarGCN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout, nout):
        super(SingleScalarGCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        self.scalar = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(SingleScalarGraphConvolution())
        self.linear =  nn.Linear(nhid, nout)
    def forward(self, x, adjs, sampled = None):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if sampled is not None:
            x = x[sampled[0]]
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[idx], self.scalar))
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs, self.scalar))
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
    def forward(self, x, adjs, sampled = None):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if sampled is not None:
            x = x[sampled[0]]
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs[idx]))
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(self.gcs[idx](x, adjs))
        return self.linear(x)

# Same as ScalarGCN but without initial feature transformation
class ScalarGCNNoFeatureTrans(nn.Module):
    def __init__(self, nfeat, layers, dropout, nout):
        super(ScalarGCNNoFeatureTrans, self).__init__()
        self.layers = layers
        self.gcs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(layers):
            self.gcs.append(ScalarGraphConvolution())
        self.linear =  nn.Linear(nfeat, nout)
    def forward(self, x, adjs, sampled = None):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        if sampled is not None:
            x = x[sampled[0]]
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
        # Each adjs is the ith power of the lap_matrix
        concat = torch.tensor([]).to(torch.device("cuda:0"))
        for idx in range(len(self.gcs)):
            out = self.dropout(self.gcs[idx](x, adjs[idx]))
            concat = torch.cat((concat, out), dim=1)
        x = self.dropout(F.elu(concat))
        return self.linear(x)

# Full attention
# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features, out_features, dropout, alpha):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha

#         self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         self.a = nn.Parameter(torch.FloatTensor(2*out_features, 1))
#         self.init_weights()

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def init_weights(self):
#         nn.init.xavier_uniform_(self.W.data)
#         nn.init.xavier_uniform_(self.a.data)

#     def forward(self, h, adj):
#         Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features=D)
#         N = Wh.size(0)

#         a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         # attention = F.dropout(attention, self.dropout)
#         h_prime = torch.mm(attention, Wh)

#         return F.elu(h_prime)

# Split into a1 and a2
# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features, out_features, dropout, alpha):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha

#         self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         self.a_src = nn.Parameter(torch.FloatTensor(out_features, 1))
#         self.a_dest = nn.Parameter(torch.FloatTensor(out_features, 1))
#         self.init_weights()

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def init_weights(self):
#         nn.init.xavier_uniform_(self.W.data)
#         nn.init.xavier_uniform_(self.a_src.data)
#         nn.init.xavier_uniform_(self.a_dest.data)

#     def forward(self, h, adj, from_feat = None, to_feat = None):
#         Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features=D)
#         N = Wh.size(0)

#         # Compute attention coefficients using efficient broadcasting
#         f_1 = torch.matmul(Wh, self.a_src).squeeze(1)
#         f_2 = torch.matmul(Wh, self.a_dest).squeeze(1)
#         e = f_1.unsqueeze(1) + f_2.unsqueeze(0)
#         e = F.leaky_relu(e, negative_slope=self.alpha)

#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         # attention = F.dropout(attention, self.dropout)
        
#         h_prime = torch.matmul(attention, Wh)
#         return F.elu(h_prime)
    
# Use sparse_csr_tensor
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a_src = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.a_dest = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.init_weights()

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def init_weights(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a_src.data)
        nn.init.xavier_uniform_(self.a_dest.data)

    def forward(self, h, adj, from_feat = None, to_feat = None):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features=D)
        N = Wh.size(0)

        # # Compute attention coefficients using efficient broadcasting
        f_1 = torch.matmul(Wh, self.a_src).squeeze(1)
        f_2 = torch.matmul(Wh, self.a_dest).squeeze(1)

        # convert from sparse CSR tensor to sparse COO tensor
        adj = adj.to_sparse()
        indices = adj.indices()

        # row_indices, col_indices should have shape nnz. This is equivalent to
        # iterating over each adj nnz and summing the appropriate scores
        new_vals = f_1[indices[0]] + f_2[indices[1]]
        new_vals = self.leakyrelu(new_vals)

        # TODO: for sampling we can modify the adj_matrix in-place
        # TODO: could keep as CSR and write your own torch sparse softmax
        attention = torch.sparse_coo_tensor(
            indices=indices,
            values=new_vals,
            size=adj.shape,
            device="cuda:0"
        )
        attention = torch.sparse.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout)
        
        h_prime = torch.sparse.mm(attention, Wh)
        return F.elu(h_prime)
    
# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features, out_features, dropout, alpha):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha

#         self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         self.a_src = nn.Parameter(torch.FloatTensor(out_features, 1))
#         self.a_dest = nn.Parameter(torch.FloatTensor(out_features, 1))
#         self.init_weights()

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def init_weights(self):
#         nn.init.xavier_uniform_(self.W.data)
#         nn.init.xavier_uniform_(self.a_src.data)
#         nn.init.xavier_uniform_(self.a_dest.data)

#     def forward(self, h, adj, from_feat = None, to_feat = None):
#         Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features=D)
#         N = Wh.size(0)

#         # # Compute attention coefficients using efficient broadcasting
#         f_1 = torch.matmul(Wh, self.a_src).squeeze(1)
#         f_2 = torch.matmul(Wh, self.a_dest).squeeze(1)

#         # convert from sparse CSR tensor to sparse COO tensor
#         # adj = adj.to_sparse()
#         # indices = adj.indices()

#         # row_indices, col_indices should have shape nnz. This is equivalent to
#         # iterating over each adj nnz and summing the appropriate scores
#         # new_vals = f_1[indices[0]] + f_2[indices[1]]
#         # new_vals = self.leakyrelu(new_vals)

#         crow_indices = adj.crow_indices()
#         col_indices = adj.col_indices()
#         num_rows = crow_indices.size(0) - 1
#         # Compute the number of non-zero elements per row
#         row_counts = crow_indices[1:] - crow_indices[:-1]
#         row_indices = torch.repeat_interleave(torch.arange(num_rows).to("cuda:0"), row_counts)

#         new_vals = f_1[row_indices] + f_2[col_indices]

#         # TODO: for sampling we can modify the adj_matrix in-place
#         # TODO: could keep as CSR and write your own torch sparse softmax
#         # attention = torch.sparse_coo_tensor(
#         #     indices=indices,
#         #     values=new_vals,
#         #     size=adj.shape,
#         #     device="cuda:0"
#         # )
#         attention = torch.sparse_csr_tensor(
#             crow_indices = crow_indices,
#             col_indices = col_indices,
#             values = new_vals,
#             size = adj.shape,
#             device = "cuda:0"
#         )
#         # attention = torch.sparse.softmax(attention, dim=1)
#         # attention = F.dropout(attention, self.dropout)
        
#         h_prime = torch.sparse.mm(attention, Wh)
#         return F.elu(h_prime)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.gcs = nn.ModuleList()
        # Initial layer
        self.gcs.append(nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]))
        # Hidden layers
        for i in range(layers-1):
            self.gcs.append(nn.ModuleList([GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]))
        # Output transformation
        self.linear = nn.Linear(nhid * nheads, nout)

    def forward(self, feat_data, adjs, sampled = None):
        if sampled is not None:
            x = x[sampled[0]]
            for idx in range(len(self.gcs)):
                x = self.dropout(torch.cat([att(x, adjs[idx], feat_data[sampled[idx+1]], feat_data[sampled[idx]]) for att in self.gcs[idx]], dim=1))
        else:
            # adjs = adjs.to_dense()
            x = feat_data
            for idx in range(len(self.gcs)):
                x = self.dropout(torch.cat([att(x, adjs) for att in self.gcs[idx]], dim=1))
        return self.linear(x)
    

####################################################################################
####################################################################################
####################################################################################

# Use sparse_csr_tensor
class ScalarGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(ScalarGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.scalar = nn.Parameter(torch.ones(1))
        self.a_src = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.a_dest = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.init_weights()

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def init_weights(self):
        nn.init.xavier_uniform_(self.a_src.data)
        nn.init.xavier_uniform_(self.a_dest.data)

    def forward(self, h, adj, from_feat = None, to_feat = None):
        Wh = torch.mul(h, self.scalar) # h.shape: (N, in_features), Wh.shape: (N, out_features=D)
        N = Wh.size(0)

        # Compute attention coefficients using efficient broadcasting
        f_1 = torch.matmul(Wh, self.a_src).squeeze(1)
        f_2 = torch.matmul(Wh, self.a_dest).squeeze(1)

        # convert from sparse CSR tensor to sparse COO tensor
        adj = adj.to_sparse()
        indices = adj.indices()

        # row_indices, col_indices should have shape nnz. This is equivalent to
        # iterating over each adj nnz and summing the appropriate scores
        new_vals = f_1[indices[0]] + f_2[indices[1]]
        new_vals = self.leakyrelu(new_vals)

        # TODO: for sampling we can modify the adj_matrix in-place
        # TODO: could keep as CSR and write your own torch sparse softmax
        attention = torch.sparse_coo_tensor(
            indices=indices,
            values=new_vals,
            size=adj.shape,
            device="cuda:0"
        )
        attention = torch.sparse.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout)
        
        h_prime = torch.sparse.mm(attention, Wh)
        return F.elu(h_prime)

class ScalarGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads):
        super(ScalarGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.gcs = nn.ModuleList()
        # Initial layer
        self.gcs.append(nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]))
        # Hidden layers
        for i in range(layers-1):
            self.gcs.append(nn.ModuleList([ScalarGraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]))
        # Output transformation
        self.linear = nn.Linear(nhid * nheads, nout)

    def forward(self, feat_data, adjs, sampled = None):
        if sampled is not None:
            x = x[sampled[0]]
            for idx in range(len(self.gcs)):
                x = self.dropout(torch.cat([att(x, adjs[idx], feat_data[sampled[idx+1]], feat_data[sampled[idx]]) for att in self.gcs[idx]], dim=1))
        else:
            # adjs = adjs.to_dense()
            x = feat_data
            for idx in range(len(self.gcs)):
                x = self.dropout(torch.cat([att(x, adjs) for att in self.gcs[idx]], dim=1))
        return self.linear(x)