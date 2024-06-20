#!/usr/bin/env python
# coding: utf-8


from utils import *
from samplers import *
from models import *
import argparse
import multiprocessing as mp
import gc
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

init_time = time.time()

parser = argparse.ArgumentParser(description='Training GCN models')

'''
    Dataset arguments
'''
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset name: cora/citeSeer/pubmed/ppi/flickr/actor')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--n_epoch', type=int, default= 100,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default= 10,
                    help='Number of Pool')
parser.add_argument('--batch_num', type=int, default= 10,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=5,
                    help='Number of GCN layers')
parser.add_argument('--n_iters', type=int, default=1,
                    help='Number of iteration to run on a batch')
parser.add_argument('--n_stops', type=int, default=20,
                    help='Stop after number of epochs that f1 dont increase')
parser.add_argument('--samp_num', type=int, default=64,
                    help='Number of sampled nodes per layer')
parser.add_argument('--sampler', type=str, default='ladies',
                    help='Sampler Algorithms: full_restricted/graphsage/ladies/fastgcn/full')
parser.add_argument('--model', type=str, default='GCN',
                    help="""Model: GCN/scalarGCN/SGC/scalarSGC/GAT/parallelGAT/scalarGAT/
                    FGAT/parallelFGAT/scalarFGAT/SAFGAT/parallelSAFGAT/scalarSAFGAT/scalarSAFGATv2""")
parser.add_argument('--cuda', type=int, default=0,
                    help='Available GPU ID')
parser.add_argument('--log_final', action=argparse.BooleanOptionalAction,
                    help='log final results to file')
parser.add_argument('--log_runs', action=argparse.BooleanOptionalAction,
                    help='log run statistics to file')
parser.add_argument('--early_stopping', action=argparse.BooleanOptionalAction,
                    help='use early stopping')
parser.add_argument('--normalisation', type=str, default='sym_normalise',
                    help='what type of normalisation')
parser.add_argument('--oiter', type=int, default=1,
                    help='number of outer iterations')
parser.add_argument('--batching', type=str, default="repeat",
                    help='batch construction method')
parser.add_argument('--test_batching', type=str, default="full",
                    help='test batching method: full/sample')
parser.add_argument('--lr', type=float, default=0.001,
                    help='optimizer learning rate')
parser.add_argument('--n_heads', type=int, default=1,
                    help='num heads for GAT')
parser.add_argument('--nn_layers', type=int, default=1,
                    help='num layers per feature transformation')
parser.add_argument('--fnn_layers', type=int, default=1,
                    help='num layers per orig. feat. -> embedding for attn. transformation')
parser.add_argument('--log_memory_snapshot', action=argparse.BooleanOptionalAction,
                    help='log pytorch memory snapshot')
parser.add_argument('--improvement_threshold', type=float, default=0,
                    help='valid f1 improvement threshold to stop overfitting')

args = parser.parse_args()
print(f"Args: {args}")
filename = f"{args.dataset}_{args.sampler}_{args.model}_{args.n_layers}layer_{args.batching}batch"
if args.samp_num not in [5, 64]:
    filename += f"_{args.samp_num}samp"
if args.n_heads != 1:
    filename += f"_{args.n_heads}head"
if args.nn_layers != 1:
    filename += f"_{args.nn_layers}nn"
if args.fnn_layers != 1:
    filename += f"_{args.fnn_layers}fnn"

if args.cuda != -1:
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.cuda))
    else:
        print("cuda not found!")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

if args.log_memory_snapshot:
    torch.cuda.memory._record_memory_history(
        max_entries=1000000
    )

print("=============== Experiment Settings ===============")
print(f"Dataset: {args.dataset}")
print(f"Model: {args.model}")
print(f"Sampler: {args.sampler}")
print(f"Num layers: {args.n_layers}")
print()

# List inductive datasets here
if args.dataset in ["ppi"]:
    inductive = True
else:
    inductive = False

if inductive:
    edges_train, labels_train, feat_data_train, num_classes_train, train_nodes, _, _, multilabel = load_data_pyg(args.dataset, split="train")
    edges_val, labels_val, feat_data_val, num_classes_val, _, valid_nodes, _, _ = load_data_pyg(args.dataset, split="val")
    edges_test, labels_test, feat_data_test, num_classes_test, _, _, test_nodes, _ = load_data_pyg(args.dataset, split="test")
    num_classes = max([num_classes_train, num_classes_val, num_classes_test])
    num_train_nodes = feat_data_train.shape[0]
    num_val_nodes = feat_data_val.shape[0]
    num_test_nodes = feat_data_test.shape[0]
    num_feat = feat_data_train.shape[1]
else:
    edges, labels, feat_data, num_classes, train_nodes, valid_nodes, test_nodes, multilabel = load_data_pyg(args.dataset)
    num_nodes, num_feat = feat_data.shape

if inductive:
    adj_matrix_train = get_adj(edges_train, num_train_nodes)
    adj_matrix_val = get_adj(edges_val, num_val_nodes)
    adj_matrix_test = get_adj(edges_test, num_test_nodes)
    del edges_train
    del edges_val
    del edges_test
else:
    adj_matrix = get_adj(edges, num_nodes)
    del edges

if args.normalisation == 'row_normalise':
    normalise = row_normalize
elif args.normalisation == 'sym_normalise':
    normalise = sym_normalize
else:
    raise ValueError("Unacceptable normalisation method")

if inductive:
    lap_matrix_train = normalise(adj_matrix_train + sp.eye(adj_matrix_train.shape[0]))
    lap_matrix_val = normalise(adj_matrix_val + sp.eye(adj_matrix_val.shape[0]))
    lap_matrix_test = normalise(adj_matrix_test + sp.eye(adj_matrix_test.shape[0]))
    feat_data_train = feat_data_train.astype(np.float32)
    feat_data_val = feat_data_val.astype(np.float32)
    feat_data_test = feat_data_test.astype(np.float32)
    del adj_matrix_train
    del adj_matrix_val
    del adj_matrix_test
else:
    lap_matrix = normalise(adj_matrix + sp.eye(adj_matrix.shape[0]))
    if type(feat_data) == sp.lil_matrix:
        feat_data = feat_data.todense()
    feat_data = feat_data.astype(np.float32)
    del adj_matrix

print("=============== Memory Info ===============")
# Pre-processing matrices for SGC and SIGN models
if inductive:
    if args.model == "SGC" or args.model == "scalarSGC":
        memory_before = torch.cuda.memory_allocated()
        adj_sgc_train = lap_matrix_train ** args.n_layers
        adj_sgc_train = package_mxl(sparse_mx_to_torch_sparse_tensor(adj_sgc_train), device)
        mem_adj_sgc_train = torch.cuda.memory_allocated() - memory_before
        print(f"adj_sgc_train size: {roundsize(mem_adj_sgc_train)}")

        memory_before = torch.cuda.memory_allocated()
        adj_sgc_val = lap_matrix_val ** args.n_layers
        adj_sgc_val = package_mxl(sparse_mx_to_torch_sparse_tensor(adj_sgc_val), device)
        mem_adj_sgc_val = torch.cuda.memory_allocated() - memory_before
        print(f"adj_sgc_val size: {roundsize(mem_adj_sgc_val)}")

        memory_before = torch.cuda.memory_allocated()
        adj_sgc_test = lap_matrix_test ** args.n_layers
        adj_sgc_test = package_mxl(sparse_mx_to_torch_sparse_tensor(adj_sgc_test), device)
        mem_adj_sgc_test = torch.cuda.memory_allocated() - memory_before
        print(f"adj_sgc_test size: {roundsize(mem_adj_sgc_test)}")

        print(f"total adj size: {roundsize(mem_adj_sgc_train + mem_adj_sgc_val + mem_adj_sgc_test)}")
    elif args.model == "SIGN":
        memory_before = torch.cuda.memory_allocated()
        adj_sign_train = []
        adj_sign_train.append(sparse_mx_to_torch_sparse_tensor(sp.eye(num_train_nodes, dtype=float, format='csr')))
        for i in range(1, args.n_layers + 1):
            adj_sign_train.append(sparse_mx_to_torch_sparse_tensor(lap_matrix_train ** i))
        adj_sign_train = package_mxl(adj_sign_train, device)
        mem_adj_sign_train = torch.cuda.memory_allocated() - memory_before
        print(f"adj_sign_train size: {roundsize(mem_adj_sign_train)}")

        memory_before = torch.cuda.memory_allocated()
        adj_sign_val = []
        adj_sign_val.append(sparse_mx_to_torch_sparse_tensor(sp.eye(num_val_nodes, dtype=float, format='csr')))
        for i in range(1, args.n_layers + 1):
            adj_sign_val.append(sparse_mx_to_torch_sparse_tensor(lap_matrix_val ** i))
        adj_sign_val = package_mxl(adj_sign_val, device)
        mem_adj_sign_val = torch.cuda.memory_allocated() - memory_before
        print(f"adj_sign_val size: {roundsize(mem_adj_sign_val)}")

        memory_before = torch.cuda.memory_allocated()
        adj_sign_test = []
        adj_sign_test.append(sparse_mx_to_torch_sparse_tensor(sp.eye(num_test_nodes, dtype=float, format='csr')))
        for i in range(1, args.n_layers + 1):
            adj_sign_test.append(sparse_mx_to_torch_sparse_tensor(lap_matrix_test ** i))
        adj_sign_test = package_mxl(adj_sign_test, device)
        mem_adj_sign_test = torch.cuda.memory_allocated() - memory_before
        print(f"adj_sign_test size: {roundsize(mem_adj_sign_test)}")

        print(f"total adj size: {roundsize(mem_adj_sign_train + mem_adj_sign_val + mem_adj_sign_test)}")
    elif args.sampler == "full":
        memory_before = torch.cuda.memory_allocated()
        adj_full_train = package_mxl(sparse_mx_to_torch_sparse_tensor(lap_matrix_train), device)
        mem_adj_full_train = torch.cuda.memory_allocated() - memory_before
        print(f"adj_full size: {roundsize(mem_adj_full_train)}")

        memory_before = torch.cuda.memory_allocated()
        adj_full_val = package_mxl(sparse_mx_to_torch_sparse_tensor(lap_matrix_val), device)
        mem_adj_full_val = torch.cuda.memory_allocated() - memory_before
        print(f"adj_full size: {roundsize(mem_adj_full_val)}")

        memory_before = torch.cuda.memory_allocated()
        adj_full_test = package_mxl(sparse_mx_to_torch_sparse_tensor(lap_matrix_test), device)
        mem_adj_full_test = torch.cuda.memory_allocated() - memory_before
        print(f"adj_full size: {roundsize(mem_adj_full_test)}")

        print(f"total adj size: {roundsize(mem_adj_full_train + mem_adj_full_val + mem_adj_full_test)}")

else:
    if args.model == "SGC" or args.model == "scalarSGC":
        memory_before = torch.cuda.memory_allocated()
        adj_sgc = lap_matrix ** args.n_layers
        adj_sgc = package_mxl(sparse_mx_to_torch_sparse_tensor(adj_sgc), device)
        memory_after = torch.cuda.memory_allocated()
        print(f"adj_sgc size: {roundsize(memory_after - memory_before)}")
    elif args.model == "SIGN":
        memory_before = torch.cuda.memory_allocated()
        adj_sign = []
        adj_sign.append(sparse_mx_to_torch_sparse_tensor(sp.eye(num_nodes, dtype=float, format='csr')))
        for i in range(1, args.n_layers + 1):
            adj_sign.append(sparse_mx_to_torch_sparse_tensor(lap_matrix ** i))
        adj_sign = package_mxl(adj_sign, device)
        memory_after = torch.cuda.memory_allocated()
        print(f"adj_sign size: {roundsize(memory_after - memory_before)}")
    elif args.sampler == "full":
        memory_before = torch.cuda.memory_allocated()
        adj_full = package_mxl(sparse_mx_to_torch_sparse_tensor(lap_matrix), device)
        memory_after = torch.cuda.memory_allocated()
        print(f"adj_full size: {roundsize(memory_after - memory_before)}")

memory_before = torch.cuda.memory_allocated()
if inductive:
    feat_data_train = torch.FloatTensor(feat_data_train).to(device)
    feat_data_val = torch.FloatTensor(feat_data_val).to(device)
    feat_data_test = torch.FloatTensor(feat_data_test).to(device)
else:
    feat_data = torch.FloatTensor(feat_data).to(device)
memory_after = torch.cuda.memory_allocated()
print(f"feat_data size: {roundsize(memory_after - memory_before)}")

memory_before = torch.cuda.memory_allocated()
if inductive:
    labels_train    = torch.LongTensor(labels_train).to(device)
    labels_val    = torch.LongTensor(labels_val).to(device) 
    labels_test    = torch.LongTensor(labels_test).to(device) 
else:
    labels    = torch.LongTensor(labels).to(device) 
memory_after = torch.cuda.memory_allocated()
print(f"labels size: {roundsize(memory_after - memory_before)}")

pretraining_memory = roundsize(torch.cuda.memory_allocated())
print(f"Total Pretraining memory: {pretraining_memory}")
pretraining_time = time.time() - init_time
print(f"Total Pretraining time: {round(pretraining_time, 2)}")


if args.sampler == 'ladies':
    sampler = ladies_sampler
elif args.sampler == 'fastgcn':
    sampler = fastgcn_sampler
elif args.sampler == 'full':
    sampler = None
elif args.sampler == 'graphsage':
    sampler = graphsage_sampler
elif args.sampler == 'full_restricted':
    sampler = full_sampler_restricted
else:
    raise ValueError("Unacceptable sample method")

samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])

log_train_times = []
log_valid_times = []
log_total_iters = []
log_best_epoch = []
log_max_train_memory = []
log_max_val_memory = []
log_adjs_memory = []
log_val_acc = []
log_val_f1 = []
log_val_sens = []
log_val_spec = []
log_test_acc = []
log_test_f1 = []
log_test_sens = []
log_test_spec = []

model_size = 0
total_params = 0
trainable_params = 0

for oiter in range(args.oiter):
    ## Batching
    train_batches = []
    val_batches = []
    test_batches = []

    if args.batching == "full":
        train_batches.append(train_nodes)
        val_batches.append(valid_nodes)
        test_batches.append(test_nodes)
    elif args.batching == "random":
        for _ in range(args.batch_num):
            train_batches.append(np.random.choice(train_nodes, size=min(args.batch_size, len(train_nodes)), replace=False))
        val_batches.append(torch.randperm(len(valid_nodes))[:args.batch_size])
        test_batches.append(test_nodes)
    elif args.batching == "repeat":
        for _ in range(args.batch_num):
            train_batches.append(train_nodes)
        val_batches.append(valid_nodes)
        test_batches.append(test_nodes)
    elif args.batching == "random3":
        np.random.shuffle(train_nodes)
        train_batches = np.array_split(train_nodes, len(train_nodes) // args.batch_size)
        val_batches.append(valid_nodes)
        test_batches.append(test_nodes)
    else:
        raise ValueError("Unacceptable batching type")

    if args.log_runs:
        f = open(f"results/per_epoch/{filename}_{oiter}.csv", "w+")
        f.write("epoch,train_loss,val_loss,val_f1\n")

    memory_before = torch.cuda.memory_allocated()
    # nn_layers controls the number of layers in each feature transformation NN
    if args.model == "GCN":
        model = GCN(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, nn_layers=args.nn_layers).to(device)
    elif args.model == "scalarGCN":
        model = GCN(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, scalar=True, nn_layers=args.nn_layers).to(device)
    elif args.model == "fixedScalarGCN":
        model = GCN(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, fixedScalar=True).to(device)

    elif args.model == "SGC":
        model = SGC(nfeat = num_feat, nout=num_classes, layers=args.n_layers, dropout=0.2).to(device)
    elif args.model == "scalarSGC":
        model = ScalarSGC(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, nn_layers=args.nn_layers).to(device)
    elif args.model == "SIGN":
        model = SIGN(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2).to(device)

    # for scalar models we pass scalar=True, for feature-based models we pass orig_features
    # fnn_layers controls the number of layers in each orig feature -> embedding for attention NN
    elif args.model == "GAT":
        model = GAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, nn_layers=args.nn_layers).to(device)
    elif args.model == "parallelGAT":
        model = ParallelGAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, nn_layers=args.nn_layers).to(device)
    elif args.model == "scalarGAT":
        model = ParallelGAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, scalar=True, nn_layers=args.nn_layers).to(device)

    elif args.model == "FGAT":
        model = GAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, orig_features=num_feat, nn_layers=args.nn_layers, fnn_layers=args.fnn_layers).to(device)
    elif args.model == "parallelFGAT":
        model = ParallelGAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, orig_features=num_feat, nn_layers=args.nn_layers, fnn_layers=args.fnn_layers).to(device)
    elif args.model == "scalarFGAT":
        model = ParallelGAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, orig_features=num_feat, scalar=True, nn_layers=args.nn_layers, fnn_layers=args.fnn_layers).to(device)

    elif args.model == "SAFGAT":
        model = SAFGAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, orig_features=num_feat, nn_layers=args.nn_layers, fnn_layers=args.fnn_layers).to(device)
    elif args.model == "parallelSAFGAT":
        model = ParallelGAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, single_adjacency=True, orig_features=num_feat, nn_layers=args.nn_layers, fnn_layers=args.fnn_layers).to(device)
    elif args.model == "scalarSAFGAT":
        model = ParallelGAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, single_adjacency=True, orig_features=num_feat, scalar=True, nn_layers=args.nn_layers, fnn_layers=args.fnn_layers).to(device)
    elif args.model == "scalarSAFGATv2":
        model = ParallelGAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, single_adjacency=True, orig_features=num_feat, scalar=True, safgat_merge=True, nn_layers=args.nn_layers).to(device)

    elif args.model == "ZAGAT":
        model = ZAGAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, nn_layers=args.nn_layers).to(device)
    elif args.model == "parallelZAGAT":
        model = ParallelGAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, nn_layers=args.nn_layers, zero_attention=True).to(device)
    elif args.model == "scalarZAGAT":
        model = ParallelGAT(nfeat = num_feat, nhid=args.nhid, nout=num_classes, layers=args.n_layers, dropout=0.2, alpha=0.2, nheads=args.n_heads, scalar=True, nn_layers=args.nn_layers, zero_attention=True).to(device)
    else:
        raise ValueError("Unacceptable model type")

    memory_after = torch.cuda.memory_allocated()
    model_size = roundsize(memory_after - memory_before)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = 0
    best_val_f1 = 0
    best_val_sens = 0
    best_val_spec = 0
    best_epoch = 0
    cnt = 0
    train_times = []
    valid_times = []
    total_iters = 0
    print('-' * 10)

    max_train_memory_allocated = 0
    max_val_memory_allocated = 0
    max_adj_memory_allocated = 0

    for epoch in np.arange(args.n_epoch):
        model.train()
        train_losses = []
        adjs_memorys = []

        torch.cuda.reset_peak_memory_stats()
        ## Training
        if inductive:
            feat_data = feat_data_train
            labels = labels_train
            num_nodes = num_train_nodes
        for train_batch in train_batches:
            train_time_0 = time.time()
            # full batch
            if args.sampler == "full":
                if args.model == "SGC" or args.model == "scalarSGC":
                    adjs_train = adj_sgc_train if inductive else adj_sgc
                elif args.model == "SIGN":
                    adjs_train = adj_sign_train if inductive else adj_sign
                else:
                    adjs_train = adj_full_train if inductive else adj_full
                output_train = model.forward(feat_data, adjs_train)
                output_train = output_train[train_batch]
            # sampling
            else:
                if args.model == "SGC" or args.model == "scalarSGC" or args.model == "SIGN":
                    raise ValueError("SGC/SIGN must use full sampler")
                if inductive:
                    lap_matrix = lap_matrix_train

                memory_before = torch.cuda.memory_allocated()
                adjs_train, sampled = sampler(np.random.randint(2**32 - 1), train_batch, samp_num_list, num_nodes, lap_matrix, args.n_layers)
                adjs_train = package_mxl(adjs_train, device)
                memory_after = torch.cuda.memory_allocated()
                max_adj_memory_allocated = max(max_adj_memory_allocated, memory_after - memory_before)

                output_train = model.forward(feat_data, adjs_train, sampled)

            if multilabel:
                loss_train = F.binary_cross_entropy_with_logits(output_train, labels[train_batch].float())
            else:
                loss_train = F.cross_entropy(output_train, labels[train_batch])
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()
            optimizer.zero_grad()
            train_losses += [loss_train.item()]
            train_times += [time.time() - train_time_0]
        max_train_memory_allocated = max(max_train_memory_allocated, torch.cuda.max_memory_allocated())

        ## Validation
        model.eval()
        valid_losses = []
        valid_f1s = []
        if inductive:
            feat_data = feat_data_val
            labels = labels_val
            num_nodes = num_val_nodes
        for val_batch in val_batches:
            torch.cuda.reset_peak_memory_stats()
            valid_time_0 = time.time()
            if args.sampler == "full":
                if args.model == "SGC" or args.model == "scalarSGC":
                    adjs_val = adj_sgc_val if inductive else adj_sgc
                elif args.model == "SIGN":
                    adjs_val = adj_sign_val if inductive else adj_sign
                else:
                    adjs_val = adj_full_val if inductive else adj_full
                with torch.no_grad():
                    output_val = model.forward(feat_data, adjs_val)
                output_val = output_val[val_batch]
            else:
                if inductive:
                    lap_matrix = lap_matrix_val
                adjs_val, sampled = sampler(np.random.randint(2**32 - 1), val_batch, samp_num_list, num_nodes, lap_matrix, args.n_layers)
                adjs_val = package_mxl(adjs_val, device)
                with torch.no_grad():
                    output_val = model.forward(feat_data, adjs_val, sampled)
            valid_times += [time.time() - valid_time_0]
            max_val_memory_allocated = max(max_val_memory_allocated, torch.cuda.max_memory_allocated())

            if multilabel:
                loss_valid = F.binary_cross_entropy_with_logits(output_val, labels[val_batch].float())
                valid_acc, valid_f1, valid_sens, valid_spec = metrics(output_val.cpu(), labels[val_batch].cpu())
            else:
                loss_valid = F.cross_entropy(output_val, labels[val_batch])
                valid_acc, valid_f1, valid_sens, valid_spec = metrics(output_val.cpu(), labels[val_batch].cpu())
            valid_losses += [loss_valid.item()]
            
            valid_f1s += [valid_f1]

        ## Logging
        print(("Epoch: %d (%.1fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f") % (epoch, np.sum(train_times), np.average(train_losses), loss_valid, valid_f1))
        if args.log_runs:
            f.write(f"{epoch},{np.average(train_losses)},{np.average(valid_losses)},{np.average(valid_f1s)}\n")
        if valid_f1 > best_val_f1 + args.improvement_threshold:
            best_val_acc = valid_acc
            best_val_f1 = valid_f1
            best_val_sens = valid_sens
            best_val_spec = valid_spec
            best_epoch = epoch
            torch.save(model, './save/best_model.pt')
            cnt = 0
        else:
            cnt += 1
        total_iters += 1
        if args.early_stopping and cnt == args.n_stops:
            break

    ## Testing
    best_model = torch.load('./save/best_model.pt')
    best_model.eval()
    test_accs = []
    test_f1s = []
    test_senss = []
    test_specs = []

    if inductive:
        feat_data = feat_data_test
        labels = labels_test
        num_nodes = num_test_nodes
        lap_matrix = lap_matrix_test
    for test_batch in test_batches:
        if args.test_batching == "full":
            # full-batch for test normally outperform sampling
            if args.model == "SGC" or args.model == "scalarSGC":
                adjs_test = adj_sgc_test if inductive else adj_sgc
            elif args.model == "SIGN":
                adjs_test = adj_sign_test if inductive else adj_sign
            elif args.sampler == "full":
                adjs_test = adj_full_test if inductive else adj_full
            else:
                adjs_test = package_mxl(sparse_mx_to_torch_sparse_tensor(lap_matrix), device)
            with torch.no_grad():
                output_test = best_model.forward(feat_data, adjs_test)
            output_test = output_test[test_batch]
        elif args.test_batching == "sample":
            if inductive:
                lap_matrix = lap_matrix_test
            adjs_test, sampled = sampler(np.random.randint(2**32 - 1), test_batch, samp_num_list, num_nodes, lap_matrix, args.n_layers)
            adjs_test = package_mxl(adjs_test, device)
            with torch.no_grad():
                output_test = model.forward(feat_data, adjs_test, sampled)
        else:
            raise ValueError("Unacceptable test_batching method")
        test_acc, test_f1, test_sens, test_spec = metrics(output_test.cpu(), labels[test_batch].cpu())
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_senss.append(test_sens)
        test_specs.append(test_spec)

    ## Epoch-level stats
    log_train_times.append(np.sum(train_times))
    log_valid_times.append(np.sum(valid_times))
    log_total_iters.append(total_iters)
    log_best_epoch.append(best_epoch)
    log_max_train_memory.append(roundsize(max_train_memory_allocated))
    log_max_val_memory.append(roundsize(max_val_memory_allocated))
    log_adjs_memory.append(roundsize(max_adj_memory_allocated))
    log_val_acc.append(best_val_acc)
    log_val_f1.append(best_val_f1)
    log_val_sens.append(best_val_sens)
    log_val_spec.append(best_val_spec)
    log_test_acc.append(np.average(test_accs))
    log_test_f1.append(np.average(test_f1s))
    log_test_sens.append(np.average(test_senss))
    log_test_spec.append(np.average(test_specs))
    
    print('Iteration: %d, Test F1: %.3f, Best Epoch: %d' % (oiter, np.average(test_f1), best_epoch))

    # Deallocating memory to ensure next experiment is fair
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.synchronize()
    del model
    del optimizer
    del best_model
    del adjs_train
    del adjs_val
    del adjs_test
    del loss_train
    del loss_valid
    del output_train
    del output_val
    del output_test
    gc.collect()
    torch.cuda.empty_cache() 

    # OOT stopping if avg train epoch more than 12s
    if np.sum(train_times) / total_iters > 12:
        print("Out Of Time Error")
        f2 = open(f"results/final.csv", "a+")
        f2.write(filename + "\n")
        f2.write("OOT" + "\n")
        quit()

    if args.log_memory_snapshot:
        try:
            torch.cuda.memory._dump_snapshot(f"mem_log.pickle")
        except Exception as e:
            raise ValueError(f"Failed to capture memory snapshot {e}")

total_time = time.time() - init_time

print_report(args, pretraining_memory, pretraining_time, total_time, 
    model_size, total_params, trainable_params, log_train_times, log_valid_times, 
    log_total_iters, log_best_epoch, log_max_train_memory, log_max_val_memory, log_adjs_memory,
    log_val_acc, log_val_f1, log_val_sens, log_val_spec,
    log_test_acc, log_test_f1, log_test_sens, log_test_spec)

if args.log_final:
    f2 = open(f"results/final.csv", "a+")
    f2.write(filename + "\n")
    f2.write(f"{round(pretraining_memory, 2)}, {round(pretraining_time, 2)}, {round(total_time, 2)}, "
             f"{round(model_size, 2)}, {total_params}, {trainable_params}, "
            f"{mean_and_std(log_train_times)}, {mean_and_std(log_valid_times)}, "
            f"{mean_and_std(log_total_iters)}, {mean_and_std(log_best_epoch)}, "
            f"{mean_and_std(np.array(log_train_times) / np.array(log_total_iters), 3)}, "
            f"{mean_and_std(log_max_train_memory)}, {mean_and_std(log_max_val_memory)}, {mean_and_std(log_adjs_memory, 3)}, "
            f"{mean_and_std(log_val_acc, 3)}, {mean_and_std(log_val_f1, 3)}, "
            f"{mean_and_std(log_val_sens, 3)}, {mean_and_std(log_val_spec, 3)}, "
            f"{mean_and_std(log_test_acc, 3)}, {mean_and_std(log_test_f1, 3)}, "
            f"{mean_and_std(log_test_sens, 3)}, {mean_and_std(log_test_spec, 3)}\n")

if args.log_memory_snapshot:
    torch.cuda.memory._record_memory_history(enabled=None)