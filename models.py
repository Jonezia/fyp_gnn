import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from utils import *

class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True, scalar=False, fixedScalar=False, nn_layers=1):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.scalar = scalar
        self.fixedScalar = fixedScalar
        self.nn_layers = nn_layers
        if scalar:
            self.s = nn.Parameter(torch.ones(1))
        elif fixedScalar:
            pass
        else:
            if nn_layers == 1:
                self.linear = nn.Linear(n_in, n_out, bias=bias)
            else:
                self.linear = nn.Sequential()
                self.linear.add_module("linear0", nn.Linear(n_in, n_out, bias=bias))
                for i in range(1, nn_layers):
                    self.linear.add_module(f"activation{i-1}", nn.ReLU())
                    self.linear.add_module(f"linear{i}", nn.Linear(n_out, n_out, bias=bias))
    def forward(self, x, adj):
        if self.fixedScalar:
            x = torch.mul(x, 1)
        elif self.scalar:
            x = torch.mul(x, self.s)
        else:
            x = self.linear(x)
        return F.elu(torch.spmm(adj, x))

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout, nout, scalar=False, fixedScalar=False, nn_layers=1):
        super(GCN, self).__init__()
        assert(not(scalar and fixedScalar))
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid,  nhid, scalar=scalar, fixedScalar=fixedScalar, nn_layers=nn_layers))
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
# aggregate in n_feat and transform n_feat -> n_out
class ScalarGCNNoUpTrans(nn.Module):
    def __init__(self, nfeat, layers, dropout, nout):
        super(ScalarGCNNoUpTrans, self).__init__()
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
    
# Same as ScalarGCN but without final feature transformation
# transform to n_out and aggregate
class ScalarGCNNoDownTrans(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout, nout):
        super(ScalarGCNNoDownTrans, self).__init__()
        self.layers = layers
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nout))
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
        return x
    
class SGC(nn.Module):
    def __init__(self, nfeat, layers, dropout, nout):
        super(SGC, self).__init__()
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.linear =  nn.Linear(nfeat, nout)
    def forward(self, x, adj_k):
        x = self.dropout(torch.spmm(adj_k, x))
        return self.linear(x)
    
class ScalarSGC(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout, nout, bias=True, nn_layers=1):
        super(ScalarSGC, self).__init__()
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        if nn_layers == 1:
            self.w = nn.Linear(nfeat, nhid)
        else:
            self.w = nn.Sequential()
            self.w.add_module("linear0", nn.Linear(nfeat, nhid, bias=bias))
            for i in range(1, nn_layers):
                self.w.add_module(f"activation{i-1}", nn.ReLU())
                self.w.add_module(f"linear{i}", nn.Linear(nhid, nhid, bias=bias))
        self.linear =  nn.Linear(nhid, nout)
    def forward(self, x, adj_k):
        x = self.w(x)
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
    
# Use sparse_csr_tensor
class GraphAttentionHead(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, scalar=False):
        super(GraphAttentionHead, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.scalar = scalar

        if self.scalar:
            self.s = nn.Parameter(torch.ones(1))
        else:
            self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a_src = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.a_dest = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.init_weights()

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def init_weights(self):
        if not self.scalar:
            nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a_src.data)
        nn.init.xavier_uniform_(self.a_dest.data)

    def forward(self, h, adj, from_feat = None, to_feat = None):
        if self.scalar:
            Wh = torch.mul(h, self.s)
        else:
            Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features=D)
        N = Wh.size(0)

        # # Compute attention coefficients using efficient broadcasting
        f_1 = torch.matmul(Wh, self.a_src).squeeze(1)
        f_2 = torch.matmul(Wh, self.a_dest).squeeze(1)

        # convert from sparse CSR tensor to sparse COO tensor
        # print(adj)
        adj = adj.to_sparse()
        # print(adj)
        indices = adj.indices()

        # row_indices, col_indices should have shape nnz. This is equivalent to
        # iterating over each adj nnz and summing the appropriate scores
        new_vals = f_1[indices[0]] + f_2[indices[1]]
        new_vals = self.leakyrelu(new_vals)

        # TODO: could keep as CSR and write your own torch sparse softmax
        attention = torch.sparse_coo_tensor(
            indices=indices,
            values=new_vals,
            size=adj.shape,
            device="cuda:0"
        )
        attention = torch.sparse.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout)
        
        mem_before = torch.cuda.memory_allocated()
        h_prime = torch.sparse.mm(attention, Wh)
        mem_after = torch.cuda.memory_allocated()
        # print(f"mem: {roundsize(mem_after)}")
        # print(f"mem diff: {roundsize(mem_after - mem_before)}")

        # m, n = adj.shape
        # mem_before = torch.cuda.memory_allocated()
        # h_prime = torch_sparse.spmm(indices, new_vals, m, n, Wh)
        # mem_after = torch.cuda.memory_allocated()
        # print(f"mem: {roundsize(mem_after)}")
        # print(f"mem diff: {roundsize(mem_after - mem_before)}")
        return F.elu(h_prime)
    
# Use sparse_csr_tensor
class fGraphAttentionHead(nn.Module):
    def __init__(self, in_features, out_features, orig_features, dropout, alpha):
        super(fGraphAttentionHead, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.orig_features = orig_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.fW = nn.Parameter(torch.FloatTensor(orig_features, out_features))
        self.a_src = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.a_dest = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.init_weights()

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def init_weights(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.fW.data)
        nn.init.xavier_uniform_(self.a_src.data)
        nn.init.xavier_uniform_(self.a_dest.data)

    def forward(self, h, adj, from_feat = None, to_feat = None):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features=D)
        N = Wh.size(0)

        # # Compute attention coefficients using efficient broadcasting
        # with sampling
        if from_feat is not None and to_feat is not None:
            h_from = torch.mm(from_feat, self.fW)
            h_to = torch.mm(to_feat, self.fW)
            f_1 = torch.matmul(h_from, self.a_src).squeeze(1)
            f_2 = torch.matmul(h_to, self.a_dest).squeeze(1)
        # with full/no sampling
        else:
            h_from = torch.mm(from_feat, self.fW)
            f_1 = torch.matmul(h_from, self.a_src).squeeze(1)
            f_2 = torch.matmul(h_from, self.a_dest).squeeze(1)

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

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.gcs = nn.ModuleList()
        # Initial layer
        self.gcs.append(nn.ModuleList([GraphAttentionHead(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]))
        # Hidden layers
        for i in range(layers-1):
            self.gcs.append(nn.ModuleList([GraphAttentionHead(nhid * nheads, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]))
        # Output transformation
        self.linear = nn.Linear(nhid * nheads, nout)

    def forward(self, x, adjs, sampled = None):
        if sampled is not None:
            raise ValueError("default GATs cannot be combined with sampling, use fGAT or laGATs")
        else:
            for idx in range(len(self.gcs)):
                x = self.dropout(torch.cat([att(x, adjs) for att in self.gcs[idx]], dim=1))
        return self.linear(x)
    
class fGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads):
        super(fGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.gcs = nn.ModuleList()
        # Initial layer
        self.gcs.append(nn.ModuleList([fGraphAttentionHead(nfeat, nhid, nfeat, dropout=dropout, alpha=alpha) for _ in range(nheads)]))
        # Hidden layers
        for i in range(layers-1):
            self.gcs.append(nn.ModuleList([fGraphAttentionHead(nhid * nheads, nhid, nfeat, dropout=dropout, alpha=alpha) for _ in range(nheads)]))
        # Output transformation
        self.linear = nn.Linear(nhid * nheads, nout)

    def forward(self, feat_data, adjs, sampled = None):
        if sampled is not None:
            x = feat_data[sampled[0]]
            for idx in range(len(self.gcs)):
                x = self.dropout(torch.cat([att(x, adjs[idx], feat_data[sampled[idx+1]], feat_data[sampled[idx]]) for att in self.gcs[idx]], dim=1))
        else:
            x = feat_data
            for idx in range(len(self.gcs)):
                x = self.dropout(torch.cat([att(x, adjs, feat_data) for att in self.gcs[idx]], dim=1))
        return self.linear(x)
    
# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features, out_features, nheads, dropout, alpha):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.nheads = nheads
#         self.out_features = out_features
#         self.alpha = alpha

#         # We can treat this as nheads independent weight matrices of shape (in_features, out_features)
#         self.W = nn.Parameter(torch.FloatTensor(in_features, nheads * out_features))
#         # Likewise these are parallel scoring functions for each head
#         self.a_src = nn.Parameter(torch.FloatTensor(nheads, out_features, 1))
#         self.a_dest = nn.Parameter(torch.FloatTensor(nheads, out_features, 1))
#         self.init_weights()

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def init_weights(self):
#         nn.init.xavier_uniform_(self.W.data)
#         nn.init.xavier_uniform_(self.a_src.data)
#         nn.init.xavier_uniform_(self.a_dest.data)

#     def forward(self, h, adj, from_feat = None, to_feat = None):
#         # h.shape: (N, in_features), Wh.shape: (N, nheads=M x out_features=D) -> (N, M, D)
#         Wh = torch.mm(h, self.W).view(-1, self.nheads, self.out_features)
#         N = Wh.size(0)

#         # # Compute attention coefficients using efficient broadcasting
#         # (N, M, D) x (M, D, 1) -> (N, D, 1)
#         f_1 = torch.matmul(Wh, self.a_src)
#         f_2 = torch.matmul(Wh, self.a_dest)

#         # convert from sparse CSR tensor to sparse COO tensor
#         adj = adj.to_sparse()
#         indices = adj.indices()

#         # row_indices, col_indices should have shape nnz. This is equivalent to
#         # iterating over each adj nnz and summing the appropriate scores
#         new_vals = f_1[indices[0]] + f_2[indices[1]]
#         new_vals = self.leakyrelu(new_vals)

#         # TODO: for sampling we can modify the adj_matrix in-place
#         # TODO: could keep as CSR and write your own torch sparse softmax
#         attention = torch.sparse_coo_tensor(
#             indices=indices,
#             values=new_vals,
#             size=adj.shape,
#             device="cuda:0"
#         )
#         attention = torch.sparse.softmax(attention, dim=1)
#         # attention = F.dropout(attention, self.dropout)
        
#         h_prime = torch.sparse.mm(attention, Wh)
#         return F.elu(h_prime)

# # GAT merged implementation
# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads):
#         super(GAT, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.layers = layers
#         self.gcs = nn.ModuleList()
#         # Initial layer
#         self.gcs.append(GraphAttentionLayer(nfeat, nhid, nheads, dropout=dropout, alpha=alpha))
#         # Hidden layers
#         for i in range(layers-1):
#             self.gcs.append(GraphAttentionLayer(nhid * nheads, nhid, nheads, dropout=dropout, alpha=alpha))
#         # Output transformation
#         self.linear = nn.Linear(nhid * nheads, nout)

#     def forward(self, feat_data, adjs, sampled = None):
#         if sampled is not None:
#             x = feat_data[sampled[0]]
#             for idx in range(len(self.gcs)):
#                 x = self.dropout(self.gcs[idx](x, adjs[idx], feat_data[sampled[idx+1]], feat_data[sampled[idx]]))
#         else:
#             x = feat_data
#             for idx in range(len(self.gcs)):
#                 x = self.dropout(self.gcs[idx](x, adjs))
#         return self.linear(x)
    
# class SharedGAT(nn.Module):
#     def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads):
#         super(SharedGAT, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.layers = layers
#         self.gcs = nn.ModuleList()
#         self.ws = nn.ParameterList()
#         # Initial layer
#         self.gcs.append(nn.ModuleList([SharedGraphAttentionHead(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]))
#         self.ws.append(nn.Parameter(torch.FloatTensor(nfeat, nhid)))
#         # Hidden layers
#         for i in range(layers-1):
#             self.gcs.append(nn.ModuleList([SharedGraphAttentionHead(nhid * nheads, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]))
#             self.ws.append(nn.Parameter(torch.FloatTensor(nhid * nheads, nhid)))
#         self.init_weights()
#         # Output transformation
#         self.linear = nn.Linear(nhid * nheads, nout)

#     def init_weights(self):
#         for w in self.ws:
#             nn.init.xavier_uniform_(w.data)

#     def forward(self, feat_data, adjs, sampled = None):
#         if sampled is not None:
#             x = x[sampled[0]]
#             for idx in range(len(self.gcs)):
#                 x = self.dropout(torch.cat([att(x, adjs[idx], feat_data[sampled[idx+1]], feat_data[sampled[idx]]) for att in self.gcs[idx]], dim=1))
#         else:
#             x = feat_data
#             for idx in range(len(self.gcs)):
#                 x = self.dropout(torch.cat([att(x, adjs, self.ws[idx]) for att in self.gcs[idx]], dim=1))
#         return self.linear(x)

class ParallelGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads, scalar=False):
        super(ParallelGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.gcs = nn.ModuleList()
    
        for _ in range(nheads):
            head_gcs = nn.ModuleList()
            # Initial layer
            head_gcs.append(GraphAttentionHead(nfeat, nhid, dropout=dropout, alpha=alpha))
            # Hidden layers
            for i in range(layers-1):
                head_gcs.append(GraphAttentionHead(nhid, nhid, dropout=dropout, alpha=alpha, scalar=scalar))
            self.gcs.append(head_gcs)
        # Output transformation
        self.linear = nn.Linear(nhid * nheads, nout)

    def forward(self, feat_data, adjs, sampled = None):
        if sampled is not None:
            raise ValueError("default GATs cannot be combined with sampling, use fGAT")
        else:
            concat = torch.tensor([]).to(torch.device("cuda:0"))
            for head_gcs in self.gcs:
                x = feat_data
                for idx in range(len(head_gcs)):
                    x = self.dropout(head_gcs[idx](x, adjs))
                concat = torch.cat((concat, x), dim=1)
        return self.linear(concat)