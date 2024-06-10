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
            self.w = generate_tiered_model(n_in, n_out, nn_layers, bias=bias)
    def forward(self, x, adj):
        if self.fixedScalar:
            x = torch.mul(x, 1)
        elif self.scalar:
            x = torch.mul(x, self.s)
        else:
            x = self.w(x)
        return F.elu(torch.sparse.mm(adj, x))

# Generate NN that geometrically decreases dimension for initial feature transformation
def generate_tiered_model(n_in, n_out, nn_layers, bias=True):
    if nn_layers == 1:
        return nn.Linear(n_in, n_out, bias=bias)
    compression_factor = (n_in / n_out) ** (1. / (nn_layers))
    # sizes: [n_in, n_hid_1, n_hid_2, ..., n_out] (len nn_layers + 1)
    sizes = [int(n_in // (compression_factor ** i)) for i in range(nn_layers)]
    sizes.append(n_out)
    layers = []
    layers.append(nn.Linear(sizes[0], sizes[1], bias=bias))
    for i in range(1, nn_layers):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[i], sizes[i+1], bias=bias))
    return nn.Sequential(*layers)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout, nout, scalar=False, fixedScalar=False, nn_layers=1):
        super(GCN, self).__init__()
        assert(not(scalar and fixedScalar))
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat, nhid, nn_layers=nn_layers))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid, nhid, scalar=scalar, fixedScalar=fixedScalar))
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
    
class SGC(nn.Module):
    def __init__(self, nfeat, layers, dropout, nout):
        super(SGC, self).__init__()
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.linear =  nn.Linear(nfeat, nout)
    def forward(self, x, adj_k):
        x = self.dropout(torch.sparse.mm(adj_k, x))
        return self.linear(x)
    
class ScalarSGC(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout, nout, bias=True, nn_layers=1):
        super(ScalarSGC, self).__init__()
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.w = generate_tiered_model(nfeat, nhid, nn_layers, bias=bias)
        self.linear =  nn.Linear(nhid, nout)
    def forward(self, x, adj_k):
        x = self.w(x)
        x = self.dropout(torch.sparse.mm(adj_k, x))
        return self.linear(x)
    
class SIGNConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(SIGNConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=bias)
    def forward(self, x, adj):
        out = self.linear(x)
        return torch.sparse.mm(adj, out)

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


############################################################################################
#                                      GAT MODELS                                          #
############################################################################################

# Graph Attention Head used for GAT, FGAT, and their parallel/scalar versions
# Parallel/Scalar versions construct GraphAttentionStream from GraphAttentionHead
# For SAFGAT and Parallel/Scalar SAFGAT we just use Graph Convolution and feed pre-computed attention
class GraphAttentionHead(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, scalar=False, orig_features=None, nn_layers=1, fnn_layers=1):
        super(GraphAttentionHead, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.scalar = scalar
        self.orig_features = orig_features

        if self.scalar:
            self.s = nn.Parameter(torch.ones(1))
        else:
            self.W = generate_tiered_model(in_features, out_features, nn_layers)
        # If we have fed in an original features dim then we are using FGAT
        if orig_features:
            self.fW = generate_tiered_model(orig_features, out_features, fnn_layers)
        self.a_src = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.a_dest = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.init_weights()

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def init_weights(self):
        nn.init.xavier_uniform_(self.a_src.data)
        nn.init.xavier_uniform_(self.a_dest.data)

    # If from_feat is not None then we are feeding in feature info for FGAT
    # If both from_feat & to_feat are not None then we are feeding in feature info for sampled from/to nodes,
    # otherwise if just from_feat is not None then we are feeding in all the features for full batch
    def forward(self, h, adj, from_feat = None, to_feat = None):
        if self.scalar:
            Wh = torch.mul(h, self.s)
        else:
            Wh = self.W(h) # h.shape: (N, in_features), Wh.shape: (N, out_features=D)
        N = Wh.size(0)
        # Wh = self.leakyrelu(Wh)

        # FGAT
        if self.orig_features is not None:
            attention = feature_attention(adj, self.fW, self.a_src, self.a_dest, from_feat, to_feat, self.dropout, self.leakyrelu)
        # normal GAT
        else:
            if from_feat is not None or to_feat is not None:
                raise ValueError("GAT cannot use orig. features, use FGAT")
            attention = gat_attention(adj, Wh, self.a_src, self.a_dest, self.dropout, self.leakyrelu)
        
        h_prime = torch.sparse.mm(attention, Wh)
        # m, n = adj.shape
        # h_prime = torch_sparse.spmm(indices, new_vals, m, n, Wh)
        return F.elu(h_prime)
    
def gat_attention(adj, Wh, a_src, a_dest, dropout, leakyrelu):
    f_1 = torch.matmul(Wh, a_src).squeeze(1)
    f_2 = torch.matmul(Wh, a_dest).squeeze(1)
    return construct_attn_matrix(adj, f_1, f_2, dropout, leakyrelu)

def feature_attention(adj, feature_trans, a_src, a_dest, from_feat, to_feat, dropout, leakyrelu):
    # for FGAT we must either get the features of all the nodes (in from_feat)
    # or the features of the from and to nodes for the layer
    # with sampling
    if to_feat is not None:
        if feature_trans is not None:
            h_from = feature_trans(from_feat)
            h_to = feature_trans(to_feat)
        else:
            h_from = from_feat
            h_to = to_feat
        f_1 = torch.matmul(h_from, a_src).squeeze(1)
        f_2 = torch.matmul(h_to, a_dest).squeeze(1)
    # with full/no sampling
    else:
        if feature_trans is not None:
            h_from = feature_trans(from_feat)
        else:
            h_from = from_feat
        f_1 = torch.matmul(h_from, a_src).squeeze(1)
        f_2 = torch.matmul(h_from, a_dest).squeeze(1)
    return construct_attn_matrix(adj, f_1, f_2, dropout, leakyrelu)

def construct_attn_matrix(adj, f_1, f_2, dropout, leakyrelu):
    # convert from sparse CSR tensor to sparse COO tensor
    adj = adj.to_sparse()
    indices = adj.indices()

    # row_indices, col_indices should have shape nnz. This is equivalent to
    # iterating over each adj nnz and summing the appropriate scores
    new_vals = f_1[indices[0]] + f_2[indices[1]]
    new_vals = leakyrelu(new_vals)
    new_vals = F.dropout(new_vals, dropout)

    # TODO: could keep as CSR and write your own torch sparse softmax
    attention = torch.sparse_coo_tensor(
        indices=indices,
        values=new_vals,
        size=adj.shape,
        device="cuda:0"
    )
    attention = torch.sparse.softmax(attention, dim=1)
    return attention

# covers: GAT, FGAT
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads, orig_features=None, nn_layers=1, fnn_layers=1):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.gcs = nn.ModuleList()
        self.orig_features = orig_features
        # Initial layer
        self.gcs.append(nn.ModuleList([GraphAttentionHead(nfeat, nhid, dropout=dropout, alpha=alpha,
            orig_features=orig_features, nn_layers=nn_layers, fnn_layers=fnn_layers) for _ in range(nheads)]))
        # Hidden layers
        for i in range(layers-1):
            self.gcs.append(nn.ModuleList([GraphAttentionHead(nhid * nheads, nhid, dropout=dropout, alpha=alpha,
                orig_features=orig_features, fnn_layers=fnn_layers) for _ in range(nheads)]))
        # Output transformation
        self.linear = nn.Linear(nhid * nheads, nout)

    def forward(self, feat_data, adjs, sampled = None):
        # FGAT
        if self.orig_features is not None:
            # sampling
            if sampled is not None:
                x = feat_data[sampled[0]]
                for idx in range(len(self.gcs)):
                    x = self.dropout(torch.cat([att(x, adjs[idx], feat_data[sampled[idx+1]], feat_data[sampled[idx]]) for att in self.gcs[idx]], dim=1))
            else:
                x = feat_data
                for idx in range(len(self.gcs)):
                    x = self.dropout(torch.cat([att(x, adjs, feat_data) for att in self.gcs[idx]], dim=1))
            return self.linear(x)
        # GAT
        else:
            # sampling
            if sampled is not None:
                raise ValueError("default GATs cannot be combined with sampling, use FGAT")
            else:
                x = feat_data
                for idx in range(len(self.gcs)):
                    x = self.dropout(torch.cat([att(x, adjs) for att in self.gcs[idx]], dim=1))
            return self.linear(x)

class SAFGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads, orig_features, nn_layers=1, fnn_layers=1):
        super(SAFGAT, self).__init__()
        assert(orig_features is not None)
        self.dropout = nn.Dropout(dropout)
        self.dropout_raw = dropout
        self.layers = layers
        self.orig_features = orig_features
        self.nheads = nheads
        self.gcs = nn.ModuleList()
        # Because we calculate the attention outside the GraphAttentionHeads we use GraphConvolution instead
        # Initial layer
        self.gcs.append(nn.ModuleList([GraphConvolution(nfeat, nhid, nn_layers) for _ in range(nheads)]))
        # Hidden layers
        for i in range(layers-1):
            self.gcs.append(nn.ModuleList([GraphConvolution(nhid * nheads, nhid) for _ in range(nheads)]))
        # Output transformation
        self.linear = nn.Linear(nhid * nheads, nout)

        # Learning attn function for each head
        self.fWs = nn.ModuleList([generate_tiered_model(orig_features, nhid, fnn_layers) for _ in range(nheads)])
        self.a_srcs = nn.ParameterList([nn.Parameter(torch.FloatTensor(nhid, 1)) for _ in range(nheads)])
        self.a_dests = nn.ParameterList([nn.Parameter(torch.FloatTensor(nhid, 1)) for _ in range(nheads)])
        self.init_weights()

        self.leakyrelu = nn.LeakyReLU(alpha)

    def init_weights(self):
        for a_src in self.a_srcs:
            nn.init.xavier_uniform_(a_src.data)
        for a_dest in self.a_dests:
            nn.init.xavier_uniform_(a_dest.data)

    def forward(self, feat_data, adjs, sampled = None):
        # sampling
        if sampled is not None:
            x = feat_data[sampled[0]]
            # iterate over layers
            for idx in range(len(self.gcs)):
                # get from and to nodes and for layer
                from_feat = feat_data[sampled[idx+1]]
                to_feat = feat_data[sampled[idx]]
                adj = adjs[idx]
                # precalculate each head's attention for given layer
                attentions = [feature_attention(adj, self.fWs[i], self.a_srcs[i], self.a_dests[i], from_feat, to_feat, self.dropout_raw, self.leakyrelu) for i in range(self.nheads)]
                x = self.dropout(torch.cat([self.gcs[idx][head_idx](x, attentions[head_idx]) for head_idx in range(len(self.gcs[idx]))], dim=1))
        else:
            x = feat_data
            # precalculate each head's attention
            attentions = [feature_attention(adjs, self.fWs[i], self.a_srcs[i], self.a_dests[i], feat_data, None, self.dropout_raw, self.leakyrelu) for i in range(self.nheads)]
            for idx in range(len(self.gcs)):
                x = self.dropout(torch.cat([self.gcs[idx][head_idx](x, attentions[head_idx]) for head_idx in range(len(self.gcs[idx]))], dim=1))
        return self.linear(x)

# GAT that just uses adjacency matrix instead of learning any based on attention
# like a GCN except we have multiple heads with different feature mappings
class ZAGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads, nn_layers=1):
        super(ZAGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout_raw = dropout
        self.layers = layers
        self.nheads = nheads
        self.gcs = nn.ModuleList()
        # Because we calculate the attention outside the GraphAttentionHeads we use GraphConvolution instead
        # Initial layer
        self.gcs.append(nn.ModuleList([GraphConvolution(nfeat, nhid, nn_layers) for _ in range(nheads)]))
        # Hidden layers
        for i in range(layers-1):
            self.gcs.append(nn.ModuleList([GraphConvolution(nhid * nheads, nhid) for _ in range(nheads)]))
        # Output transformation
        self.linear = nn.Linear(nhid * nheads, nout)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, feat_data, adjs, sampled = None):
        # sampling
        if sampled is not None:
            x = feat_data[sampled[0]]
            # iterate over layers
            for idx in range(len(self.gcs)):
                x = self.dropout(torch.cat([self.gcs[idx][head_idx](x, adjs[idx]) for head_idx in range(len(self.gcs[idx]))], dim=1))
        else:
            x = feat_data
            for idx in range(len(self.gcs)):
                x = self.dropout(torch.cat([self.gcs[idx][head_idx](x, adjs) for head_idx in range(len(self.gcs[idx]))], dim=1))
        return self.linear(x)

# covers: ParallelGAT, ScalarGAT, ParallelFGAT, ScalarFGAT, ParallelSAFGAT, ScalarSAFGAT, ParallelZAGAT, ScalarZAGAT
# this implementation utilises head splitting
class ParallelGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nheads, scalar=False, orig_features=None,
                 single_adjacency=False, safgat_merge=False, nn_layers=1, fnn_layers=1, zero_attention=False):
        if safgat_merge:
            assert scalar and single_adjacency

        super(ParallelGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.nout = nout
        self.head_streams = nn.ModuleList()
        self.linear = nn.Linear(nhid * nheads, nout)
    
        for _ in range(nheads):
            if zero_attention:
                self.head_streams.append(GraphZeroAttentionStream(nfeat, nhid, nout, layers, dropout, alpha, nn_layers=nn_layers, scalar=scalar))
            elif single_adjacency:
                if safgat_merge:
                    # ScalarSAFGATv2
                    self.head_streams.append(ScalarSAFGATv2Stream(nfeat, nhid, nout, layers, dropout, alpha, nn_layers=nn_layers))
                else:
                    # ParallelSAFGAT / ScalarSAFGAT
                    self.head_streams.append(GraphSingleAttentionStream(nfeat, nhid, nout, layers, dropout, alpha,
                        scalar=scalar,orig_features=orig_features, nn_layers=nn_layers, fnn_layers=fnn_layers))
            else:
                # ParallelGAT / ScalarGAT / ParallelFGAT / ScalarFGAT
                self.head_streams.append(GraphAttentionStream(nfeat, nhid, nout, layers, dropout, alpha,
                    scalar=scalar, orig_features=orig_features, nn_layers=nn_layers, fnn_layers=fnn_layers))

    def forward(self, feat_data, adjs, sampled = None):
        n = feat_data.shape[0]
        concat = torch.tensor([]).to(torch.device("cuda:0"))
        for head_stream in self.head_streams:
            out = head_stream.forward(feat_data, adjs, sampled)
            concat = torch.cat((concat, out), dim=1)
        concat = self.dropout(concat)
        return self.linear(concat)

# used in: ParallelGAT, ScalarGAT, ParallelFGAT, ScalarFGAT
class GraphAttentionStream(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, scalar=False, orig_features=None, nn_layers=1, fnn_layers=1):
        super(GraphAttentionStream, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.orig_features = orig_features

        self.head_gcs = nn.ModuleList()
        # Initial layer
        self.head_gcs.append(GraphAttentionHead(nfeat, nhid, dropout=dropout, alpha=alpha,
            orig_features=orig_features, nn_layers=nn_layers, fnn_layers=fnn_layers))
        # Hidden layers
        for i in range(layers-1):
            self.head_gcs.append(GraphAttentionHead(nhid, nhid, dropout=dropout, alpha=alpha, scalar=scalar,
                orig_features=orig_features, fnn_layers=fnn_layers))

    def forward(self, feat_data, adjs, sampled = None):
        # ParallelFGAT / ScalarFGAT
        if self.orig_features is not None:
            # sampling
            if sampled is not None:
                x = feat_data[sampled[0]]
                for idx in range(len(self.head_gcs)):
                    x = self.dropout(self.head_gcs[idx](x, adjs[idx], feat_data[sampled[idx+1]], feat_data[sampled[idx]]))
            else:
                x = feat_data
                for idx in range(len(self.head_gcs)):
                    x = self.dropout(self.head_gcs[idx](x, adjs, feat_data))
        # ParallelGAT / ScalarGAT
        else:
            # sampling
            if sampled is not None:
                raise ValueError("default GATs cannot be combined with sampling, use FGAT")
            else:
                x = feat_data
                for idx in range(len(self.head_gcs)):
                    x = self.dropout(self.head_gcs[idx](x, adjs))
        return x

# used in: ParallelSAFGAT, ScalarSAFGAT
class GraphSingleAttentionStream(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, scalar=False, orig_features=None, nn_layers=1, fnn_layers=1):
        super(GraphSingleAttentionStream, self).__init__()
        # GraphSingleAttentionStreams can onl be used with feature-based attention
        assert(orig_features is not None)

        self.dropout = nn.Dropout(dropout)
        self.dropout_raw = dropout
        self.layers = layers
        self.orig_features = orig_features
        self.scalar = scalar

        self.head_gcs = nn.ModuleList()
        # Initial layer
        self.head_gcs.append(GraphConvolution(nfeat, nhid, nn_layers=nn_layers))
        # Hidden layers
        for i in range(layers-1):
            self.head_gcs.append(GraphConvolution(nhid, nhid, scalar=scalar))

        self.fW = generate_tiered_model(nfeat, nhid, nn_layers=fnn_layers)
        self.a_src = nn.Parameter(torch.FloatTensor(nhid, 1))
        self.a_dest = nn.Parameter(torch.FloatTensor(nhid, 1))
        self.init_weights()

        self.leakyrelu = nn.LeakyReLU(alpha)

    def init_weights(self):
        nn.init.xavier_uniform_(self.a_src.data)
        nn.init.xavier_uniform_(self.a_dest.data)

    # we learn a single FUNCTION which can be used to calculate pairwise attentions
    # for sampling, we don't store the calculated whole attn matrix but calculate mini attn matrices on the fly
    # for full batch, this will just recalculate the same thing so we can store the whole attn matrix
    
    def forward(self, feat_data, adjs, sampled = None):
        # the only case here is that we use feature attention
        # sampling
        if sampled is not None:
            x = feat_data[sampled[0]]
            for idx in range(len(self.head_gcs)):
                # for sampled, we use the learnt attn function to calculate mini adj per layer
                from_feat = feat_data[sampled[idx+1]]
                to_feat = feat_data[sampled[idx]]
                attention = feature_attention(adjs[idx], self.fW, self.a_src, self.a_dest, from_feat, to_feat, self.dropout_raw, self.leakyrelu)
                x = self.dropout(self.head_gcs[idx](x, attention))
        else:
            x = feat_data
            # for full, because the adj per layer is constant we calculate it once and use for all layers
            attention = feature_attention(adjs, self.fW, self.a_src, self.a_dest, feat_data, None, self.dropout_raw, self.leakyrelu)
            for idx in range(len(self.head_gcs)):
                x = self.dropout(self.head_gcs[idx](x, attention))
        return x
    
# model uses the single feature transformation of scalarisation as the feature transformation for the single attention
# i.e. instead of multipling feat_data by fW to get embeddings for the attention function, we simply
# pass the embeddings learnt after the intiial transformation phase to the attention function
class ScalarSAFGATv2Stream(nn.Module):
    # basically nn_layers acts as shared nn_layers & fnn_layers
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, nn_layers=1, bias=True):
        super(ScalarSAFGATv2Stream, self).__init__()
        # GraphSingleAttentionStreams can onl be used with feature-based attention

        self.dropout = nn.Dropout(dropout)
        self.dropout_raw = dropout
        self.layers = layers

        self.init_trans = generate_tiered_model(nfeat, nhid, nn_layers, bias=bias)

        self.head_gcs = nn.ModuleList()
        # Initial layer
        # Since we've pulled out the initial transformation, we use fixedScalar here
        self.head_gcs.append(GraphConvolution(nfeat, nhid, fixedScalar=True))
        # Hidden layers
        for i in range(layers-1):
            self.head_gcs.append(GraphConvolution(nhid, nhid, scalar=True))

        self.a_src = nn.Parameter(torch.FloatTensor(nhid, 1))
        self.a_dest = nn.Parameter(torch.FloatTensor(nhid, 1))
        self.init_weights()

        self.leakyrelu = nn.LeakyReLU(alpha)

    def init_weights(self):
        nn.init.xavier_uniform_(self.a_src.data)
        nn.init.xavier_uniform_(self.a_dest.data)
    
    def forward(self, feat_data, adjs, sampled = None):
        # the only case here is that we use feature attention
        # sampling
        if sampled is not None:
            x = feat_data[sampled[0]]
            x = self.init_trans(x)
            for idx in range(len(self.head_gcs)):
                # for sampled, we use the learnt attn function to calculate mini adj per layer
                from_feat = feat_data[sampled[idx+1]]
                to_feat = feat_data[sampled[idx]]
                # we pass init_trans as the transformation function for attention
                attention = feature_attention(adjs[idx], self.init_trans, self.a_src, self.a_dest, from_feat, to_feat, self.dropout_raw, self.leakyrelu)
                x = self.dropout(self.head_gcs[idx](x, attention))
        else:
            x = feat_data
            x = self.init_trans(x)
            # for full, because the adj per layer is constant we calculate it once and use for all layers
            # we pass the transformed features for attention calculation
            attention = feature_attention(adjs, None, self.a_src, self.a_dest, x, None, self.dropout_raw, self.leakyrelu)
            for idx in range(len(self.head_gcs)):
                x = self.dropout(self.head_gcs[idx](x, attention))
        return x
    
# used in: ParallelZAGAT, ScalarZAGAT
class GraphZeroAttentionStream(nn.Module):
    def __init__(self, nfeat, nhid, nout, layers, dropout, alpha, scalar=False, nn_layers=1):
        super(GraphZeroAttentionStream, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

        self.head_gcs = nn.ModuleList()
        # Initial layer
        self.head_gcs.append(GraphConvolution(nfeat, nhid, nn_layers=nn_layers))
        # Hidden layers
        for i in range(layers-1):
            self.head_gcs.append(GraphConvolution(nhid, nhid, scalar=scalar))

    def forward(self, feat_data, adjs, sampled = None):
        # sampling
        if sampled is not None:
            x = feat_data[sampled[0]]
            for idx in range(len(self.head_gcs)):
                x = self.dropout(self.head_gcs[idx](x, adjs[idx]))
        else:
            x = feat_data
            for idx in range(len(self.head_gcs)):
                x = self.dropout(self.head_gcs[idx](x, adjs))
        return x