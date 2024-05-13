#!/usr/bin/env python
# coding: utf-8


from utils import *
from samplers import *
from models import *
from tqdm import tqdm
import argparse
import scipy
import multiprocessing as mp

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(description='Training GCN on Cora/CiteSeer/PubMed/Reddit Datasets')

'''
    Dataset arguments
'''
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset name: Cora/CiteSeer/PubMed/PPI/Reddit')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default= 100,
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
                    help='Sampler Algorithms: ladies/fastgcn/full')
parser.add_argument('--model', type=str, default='GCN',
                    help='Model: GCN/scalarGCN/SGC')
parser.add_argument('--cuda', type=int, default=0,
                    help='Available GPU ID')
parser.add_argument('--log_final', action=argparse.BooleanOptionalAction,
                    help='log final results to file')
parser.add_argument('--log_runs', action=argparse.BooleanOptionalAction,
                    help='log run statistics to file')
parser.add_argument('--early_stopping', action=argparse.BooleanOptionalAction,
                    help='use early stopping')
parser.add_argument('--normalisation', type=str, default='row_normalise',
                    help='what type of normalisation')
parser.add_argument('--oiter', type=int, default=1,
                    help='number of outer iterations')
parser.add_argument('--batching', type=str, default="full",
                    help='batch construction method')
parser.add_argument('--test_batching', type=str, default="full",
                    help='test batching method: full/sample')
parser.add_argument('--lr', type=float, default=0.01,
                    help='optimizer learning rate')

args = parser.parse_args()

if args.cuda != -1:
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.cuda))
    else:
        print("cuda not found!")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

print("=============== Experiment Settings ===============")
print(f"Dataset: {args.dataset}")
print(f"Model: {args.model}")
print(f"Sampler: {args.sampler}")
print(f"Num layers: {args.n_layers}")
print()

edges, labels, feat_data, num_classes, train_nodes, valid_nodes, test_nodes, multilabel = load_data_pyg(args.dataset)

adj_matrix = get_adj(edges, feat_data.shape[0])

del edges

if args.normalisation == 'row_normalise':
    normalise = row_normalize
elif args.normalisation == 'sym_normalise':
    normalise = sym_normalize
else:
    raise ValueError("Unacceptable normalisation method")

lap_matrix = normalise(adj_matrix + sp.eye(adj_matrix.shape[0]))
if type(feat_data) == scipy.sparse.lil_matrix:
    print("LIL MATRIX !!!!")
    feat_data = feat_data.todense()
if feat_data.dtype == np.float64:
    print("64 BIT!!!")
feat_data = feat_data.astype(np.float32)

del adj_matrix

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
        train_batches.append(np.random.choice(train_nodes, size=args.batch_size, replace=False))
    val_batches.append(torch.randperm(len(valid_nodes))[:args.batch_size])
    test_batches.append(test_nodes)
elif args.batching == "random2":
    train_batches.append(train_nodes)
    val_batches.append(torch.randperm(len(valid_nodes))[:args.batch_size])
    test_batches.append(test_nodes)
elif args.batching == "random3":
    np.random.shuffle(train_nodes)
    train_batches = np.array_split(train_nodes, len(train_nodes) // args.batch_size)
    val_batches.append(valid_nodes)
    test_batches.append(test_nodes)
else:
    raise ValueError("Unacceptable batching type")

print("=============== Memory Info ===============")
# Pre-processing matrices for SGC and SIGN models
if args.model == "SGC":
    memory_before = torch.cuda.memory_allocated()
    adj_sgc = lap_matrix ** args.n_layers
    adj_sgc = package_mxl(sparse_mx_to_torch_sparse_tensor(adj_sgc), device)
    memory_after = torch.cuda.memory_allocated()
    print(f"adj_sgc size: {roundsize(memory_after - memory_before)}")
elif args.model == "SIGN":
    memory_before = torch.cuda.memory_allocated()
    adj_sign = []
    for i in range(args.n_layers + 1):
        adj_sign.append(sparse_mx_to_torch_sparse_tensor(lap_matrix ** i))
    adj_sign = package_mxl(adj_sign, device)
    memory_after = torch.cuda.memory_allocated()
    print(f"adj_sign size: {roundsize(memory_after - memory_before)}")
elif args.sampler == "full":
    memory_before = torch.cuda.memory_allocated()
    adj_full = package_mxl(sparse_mx_to_torch_sparse_tensor(lap_matrix), device)
    memory_after = torch.cuda.memory_allocated()
    print(f"adj_full size: {roundsize(memory_after - memory_before)}")

# Adjusting lap_matrix self-connection by epsilon for GIN
if args.model == "GIN":
    pass

memory_before = torch.cuda.memory_allocated()
feat_data = torch.FloatTensor(feat_data).to(device)
memory_after = torch.cuda.memory_allocated()
print(f"feat_data size: {roundsize(memory_after - memory_before)}")
memory_before = torch.cuda.memory_allocated()
labels    = torch.LongTensor(labels).to(device) 
memory_after = torch.cuda.memory_allocated()
print(f"labels size: {roundsize(memory_after - memory_before)}")
memory_before = torch.cuda.memory_allocated()
test_lap = package_mxl(sparse_mx_to_torch_sparse_tensor(lap_matrix), device)
memory_after = torch.cuda.memory_allocated()
print(f"full lap_matrix size: {roundsize(memory_after - memory_before)}")
del test_lap

if args.sampler == 'ladies':
    sampler = ladies_sampler
elif args.sampler == 'fastgcn':
    sampler = fastgcn_sampler
elif args.sampler == 'full':
    sampler = None
elif args.sampler == 'graphsage':
    sampler = graphsage_sampler
elif args.sampler == 'receptive_field':
    sampler = default_sampler_restricted
else:
    raise ValueError("Unacceptable sample method")

samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])


def prepare_data(pool, sampler, batches, train_nodes, valid_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    jobs = []
    for idx in batches:
        batch_nodes = train_nodes[idx]
        p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes, samp_num_list, num_nodes, lap_matrix, depth))
        jobs.append(p)
    idx = torch.randperm(len(valid_nodes))[:args.batch_size]
    batch_nodes = valid_nodes[idx]
    # get the samples for validation set: samp_num_list * 20 just sets number of samples per layer to high val to sample all nodes per layer
    p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes, samp_num_list * 20, num_nodes, lap_matrix, depth))
    jobs.append(p)
    return jobs


# pool = mp.Pool(args.pool_num)
# jobs = prepare_data(pool, sampler, batches, train_nodes, valid_nodes, samp_num_list, len(feat_data), lap_matrix, args.n_layers)

log_times = []
log_total_iters = []
log_best_epoch = []
log_max_memory = []
log_adjs_memory = []
log_test_acc = []
log_test_f1 = []
log_test_sens = []
log_test_spec = []

pre_training_max_memory = torch.cuda.max_memory_allocated()
pre_training_memory = torch.cuda.memory_allocated()
print(f"Pre training max memory: {roundsize(pre_training_max_memory)}MB")

for oiter in range(args.oiter):
    if args.log_runs:
        f = open(f"results/per_epoch/{args.dataset}_{args.sampler}_{args.model}_{args.n_layers}layer_{args.batching}batch_{oiter}.csv", "w+")
        f.write("epoch,train_loss,val_loss,val_f1\n")

    memory_before = torch.cuda.memory_allocated()
    if args.model == "GCN":
        encoder = GCN(nfeat = feat_data.shape[1], nhid=args.nhid, layers=args.n_layers, dropout = 0.2).to(device)
    elif args.model == "scalarGCN":
        encoder = ScalarGCN(nfeat = feat_data.shape[1], nhid=args.nhid, layers=args.n_layers, dropout = 0.2).to(device)
    elif args.model == "scalarSGC":
        encoder = ScalarSGC(nfeat = feat_data.shape[1], layers=args.n_layers, dropout = 0.2).to(device)
    elif args.model == "SGC":
        encoder = SGC(nfeat = feat_data.shape[1], layers=args.n_layers, dropout = 0.2).to(device)
    elif args.model == "SIGN":
        encoder = SIGN(nfeat = feat_data.shape[1], nhid=args.nhid, layers=args.n_layers, dropout = 0.2).to(device)
    else:
        raise ValueError("Unacceptable model type")
    memory_after = torch.cuda.memory_allocated()
    print(f"Encoder size: {roundsize(memory_after - memory_before)}")

    memory_before = torch.cuda.memory_allocated()
    susage  = SuGCN(encoder = encoder, num_classes=num_classes, dropout=0.5, inp = feat_data.shape[1]).to(device)
    memory_after = torch.cuda.memory_allocated()
    print(f"Susage size: {roundsize(memory_after - memory_before)}")

    pytorch_total_params = sum(p.numel() for p in susage.parameters())
    print(f"Total model parameters: {pytorch_total_params}")
    pytorch_total_params = sum(p.numel() for p in susage.parameters() if p.requires_grad)
    print(f"Trainable model parameters: {pytorch_total_params}")

    optimizer = optim.Adam(filter(lambda p : p.requires_grad, susage.parameters()), lr=args.lr)
    best_val = 0
    best_epoch = 0
    cnt = 0
    times = []
    time_0 = time.time()
    total_iters = 0
    print('-' * 10)
    max_memory_allocated = 0
    # TODO: this will just get the memory of the first adj it sees
    max_adj_memory_allocated = 0

    for epoch in np.arange(args.epoch_num):
        susage.train()
        train_losses = []
        adjs_memorys = []
        # train_data = [job.get() for job in jobs[:-1]]
        # valid_data = jobs[-1].get()
        # pool.close()
        # pool.join()
        # pool = mp.Pool(args.pool_num)
        '''
            Use CPU-GPU cooperation to reduce the overhead for sampling. (conduct sampling while training)
        '''
        # jobs = prepare_data(pool, sampler, train_batches, train_nodes, valid_nodes, samp_num_list, len(feat_data), lap_matrix, args.n_layers)

        ## Training
        torch.cuda.reset_peak_memory_stats()
        for train_batch in train_batches:
            optimizer.zero_grad()
            t1 = time.time()
            if args.sampler == "full":
                if args.model == "SGC":
                    adjs = adj_sgc
                elif args.model == "SIGN":
                    adjs = adj_sign
                else:
                    adjs = adj_full
                output = susage.forward(feat_data[train_batch], adjs)
                output = output[train_batch]
            else:
                if args.model == "SGC" or args.model == "SIGN":
                    raise ValueError("SGC/SIGN must use full sampler")
                # TODO: change this such that we can do the sampling async in the upper comment part
                memory_before = torch.cuda.memory_allocated()
                adjs, input_nodes = sampler(np.random.randint(2**32 - 1), train_batch, samp_num_list, len(feat_data), lap_matrix, args.n_layers)
                adjs = package_mxl(adjs, device)
                memory_after = torch.cuda.memory_allocated()
                max_adj_memory_allocated = max(max_adj_memory_allocated, memory_after - memory_before)
                output = susage.forward(feat_data[input_nodes], adjs)
            if multilabel:
                loss_train = F.binary_cross_entropy_with_logits(output, labels[train_batch].float())
            else:
                loss_train = F.cross_entropy(output, labels[train_batch])
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(susage.parameters(), 0.2)
            optimizer.step()
            times += [time.time() - t1]
            train_losses += [loss_train.detach().tolist()]
            del loss_train
        max_memory_allocated = torch.cuda.max_memory_allocated()

        ## Validation
        susage.eval()
        valid_losses = []
        valid_f1s = []
        for val_batch in val_batches:
            if args.sampler == "full":
                if args.model == "SGC":
                    adjs = adj_sgc
                elif args.model == "SIGN":
                    adjs = adj_sign
                else:
                    adjs = adj_full
                output = susage.forward(feat_data, adjs)
                output = output[val_batch]
            else:
                adjs, input_nodes = sampler(np.random.randint(2**32 - 1), val_batch, samp_num_list, len(feat_data), lap_matrix, args.n_layers)
                adjs = package_mxl(adjs, device)
                output = susage.forward(feat_data[input_nodes], adjs)
            if multilabel:
                loss_valid = F.binary_cross_entropy_with_logits(output, labels[val_batch].float()).detach().tolist()
                valid_f1 = f1_score((output >= 0.5).cpu().int(), labels[val_batch].cpu(), average='samples')
            else:
                loss_valid = F.cross_entropy(output, labels[val_batch]).detach().tolist()
                valid_f1 = f1_score(output.argmax(dim=1).cpu(), labels[val_batch].cpu(), average='micro')
            valid_losses += [loss_valid]
            valid_f1s += [valid_f1]

        ## Logging
        print(("Epoch: %d (%.1fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f") % (epoch, np.sum(times), np.average(train_losses), loss_valid, valid_f1))
        if args.log_runs:
            f.write(f"{epoch},{np.average(train_losses)},{np.average(valid_losses)},{np.average(valid_f1s)}\n")
        if valid_f1 > best_val + 1e-2:
            best_val = valid_f1
            best_epoch = epoch
            torch.save(susage, './save/best_model.pt')
            cnt = 0
        else:
            cnt += 1
        total_iters += 1
        if args.early_stopping and cnt == args.n_stops:
            break

    ## Testing
    time_1 = time.time()
    best_model = torch.load('./save/best_model.pt')
    best_model.eval()
    test_accs = []
    test_f1s = []
    test_senss = []
    test_specs = []

    for test_batch in test_batches:
        if args.test_batching == "full":
            # full-batch for test will always outperform sampling
            if args.model == "SGC":
                adjs = adj_sgc
            elif args.model == "SIGN":
                adjs = adj_sign
            elif args.sampler == "full":
                adjs = adj_full
            else:
                adjs = package_mxl(sparse_mx_to_torch_sparse_tensor(lap_matrix), device)
            output = susage.forward(feat_data, adjs)
            output = best_model.forward(feat_data, adjs)
            output = output[test_batch]
        elif args.test_batching == "sample":
            adjs, input_nodes = sampler(np.random.randint(2**32 - 1), test_batch, samp_num_list, len(feat_data), lap_matrix, args.n_layers)
            adjs = package_mxl(adjs, device)
            output = best_model.forward(feat_data[input_nodes], adjs)
        else:
            raise ValueError("Unacceptable test_batching method")
        test_acc, test_f1, test_sens, test_spec = metrics(output.cpu(), labels[test_batch].cpu())
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_senss.append(test_sens)
        test_specs.append(test_spec)

    ## Epoch-level stats
    log_times.append(time_1 - time_0)
    log_total_iters.append(total_iters)
    log_best_epoch.append(best_epoch)
    log_max_memory.append(roundsize(max_memory_allocated))
    log_adjs_memory.append(roundsize(max_adj_memory_allocated))
    log_test_acc.append(np.average(test_accs))
    log_test_f1.append(np.average(test_f1s))
    log_test_sens.append(np.average(test_senss))
    log_test_spec.append(np.average(test_specs))
    
    print('Iteration: %d, Test F1: %.3f, Best Epoch: %d' % (oiter, np.average(test_f1), best_epoch))

print_report(args, log_times, log_total_iters, log_best_epoch, log_max_memory, log_adjs_memory, log_test_acc, log_test_f1, log_test_sens, log_test_spec)

if args.log_final:
    f2 = open(f"results/final.csv", "a+")
    f2.write(f"{args.dataset}_{args.sampler}_{args.model}_{args.n_layers}layer_{args.batching}batch" + "\n")
    f2.write(f"{mean_and_std(log_times)}, {mean_and_std(log_total_iters)}, {mean_and_std(log_best_epoch)}, "
            f"{mean_and_std(np.array(log_times) / np.array(log_total_iters), 3)}, {mean_and_std(log_max_memory)} "
            f"{mean_and_std(log_adjs_memory)}, {mean_and_std(log_test_acc, 3)}, {mean_and_std(log_test_f1, 3)}, "
            f"{mean_and_std(log_test_sens, 3)}, {mean_and_std(log_test_spec, 3)}\n")