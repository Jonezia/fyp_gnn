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
parser.add_argument('--dataset', type=str, default='Cora',
                    help='Dataset name: Cora/CiteSeer/PubMed/PPI/Reddit')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default= 100,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default= 10,
                    help='Number of Pool')
parser.add_argument('--batch_num', type=int, default= 10,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=5,
                    help='Number of GCN layers')
parser.add_argument('--n_iters', type=int, default=1,
                    help='Number of iteration to run on a batch')
parser.add_argument('--n_stops', type=int, default=200,
                    help='Stop after number of batches that f1 dont increase')
parser.add_argument('--samp_num', type=int, default=64,
                    help='Number of sampled nodes per layer')
parser.add_argument('--sampler', type=str, default='ladies',
                    help='Sampler Algorithms: ladies/fastgcn/full')
parser.add_argument('--model', type=str, default='GCN',
                    help='Model: GCN/scalarGCN/SGC')
parser.add_argument('--cuda', type=int, default=0,
                    help='Available GPU ID')
parser.add_argument('--logging', type=bool, default=False,
                    help='log results to files')
parser.add_argument('--early_stopping', type=bool, default=True,
                    help='whether to use early stopping')
parser.add_argument('--normalisation', type=str, default='row_normalise',
                    help='what type of normalisation')
parser.add_argument('--oiter', type=int, default=1,
                    help='number of outer iterations')
parser.add_argument('--restrict_receptive_field', type=bool, default=False,
                    help='whether to restrict to receptive field')

args = parser.parse_args()


def prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    jobs = []
    for _ in process_ids:
        idx = torch.randperm(len(train_nodes))[:args.batch_size]
        batch_nodes = train_nodes[idx]
        p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes, samp_num_list, num_nodes, lap_matrix, depth))
        jobs.append(p)
    idx = torch.randperm(len(valid_nodes))[:args.batch_size]
    batch_nodes = valid_nodes[idx]
    # get the samples for validation set: samp_num_list * 20 just sets number of samples per layer to high val to sample all nodes per layer
    p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes, samp_num_list * 20, num_nodes, lap_matrix, depth))
    jobs.append(p)
    return jobs

def package_mxl(mxl, device):
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]


if args.cuda != -1:
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.cuda))
    else:
        print("cuda not found!")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
    
print(args.dataset, args.sampler)
edges, labels, feat_data, num_classes, train_nodes, valid_nodes, test_nodes, multiclass = load_data(args.dataset)

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
    feat_data = feat_data.todense()
feat_data = feat_data.astype(np.float32)

del adj_matrix

if not args.restrict_receptive_field:
    memory_before = torch.cuda.memory_allocated()
    feat_data = torch.FloatTensor(feat_data).to(device)
    memory_after = torch.cuda.memory_allocated()
    print(f"feat_data size: {roundsize(memory_after - memory_before)}")
    memory_before = torch.cuda.memory_allocated()
    labels    = torch.LongTensor(labels).to(device) 
    memory_after = torch.cuda.memory_allocated()
    print(f"labels size: {roundsize(memory_after - memory_before)}")


if args.sampler == 'ladies':
    sampler = ladies_sampler
elif args.sampler == 'fastgcn':
    sampler = fastgcn_sampler
elif args.sampler == 'full':
    sampler = default_sampler
elif args.sampler == 'receptive_field':
    sampler = default_sampler_restricted
else:
    raise ValueError("Unacceptable sample method")

process_ids = np.arange(args.batch_num)
samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])

pool = mp.Pool(args.pool_num)
jobs = prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, len(feat_data), lap_matrix, args.n_layers)

log_times = []
log_total_iters = []
log_max_memory = []
log_test_acc = []
log_test_f1 = []
log_test_sens = []
log_test_spec = []

all_res = []
for oiter in range(args.oiter):
    if args.logging:
        f = open(f"results/per_epoch/{args.dataset}_{args.sampler}_{args.model}_{args.n_layers}layer_restrict{args.restrict_receptive_field}_{oiter}.csv", "w+")
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
    else:
        raise ValueError("Unacceptable model type")
    memory_after = torch.cuda.memory_allocated()
    print(f"Encoder size: {roundsize(memory_after - memory_before)}")
    memory_before = torch.cuda.memory_allocated()
    susage  = SuGCN(encoder = encoder, num_classes=num_classes, dropout=0.5, inp = feat_data.shape[1])
    susage.to(device)
    memory_after = torch.cuda.memory_allocated()
    print(f"Susage size: {roundsize(memory_after - memory_before)}")

    pytorch_total_params = sum(p.numel() for p in susage.parameters())
    print(f"Total model parameters: {pytorch_total_params}")
    pytorch_total_params = sum(p.numel() for p in susage.parameters() if p.requires_grad)
    print(f"Trainable model parameters: {pytorch_total_params}")

    optimizer = optim.Adam(filter(lambda p : p.requires_grad, susage.parameters()))
    best_val = 0
    best_tst = -1
    cnt = 0
    times = []
    time_0 = time.time()
    total_iters = 0
    print('-' * 10)
    max_memory_allocated = 0
    for epoch in np.arange(args.epoch_num):
        susage.train()
        train_losses = []
        train_data = [job.get() for job in jobs[:-1]]
        valid_data = jobs[-1].get()
        pool.close()
        pool.join()
        pool = mp.Pool(args.pool_num)
        '''
            Use CPU-GPU cooperation to reduce the overhead for sampling. (conduct sampling while training)
        '''
        jobs = prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, len(feat_data), lap_matrix, args.n_layers)
        for _iter in range(args.n_iters):
            for adjs, input_nodes, output_nodes in train_data:
                torch.cuda.reset_peak_memory_stats()
                # memory_before = torch.cuda.memory_allocated()
                adjs = package_mxl(adjs, device)
                # memory_after = torch.cuda.memory_allocated()
                # print(f"adjs size: {roundsize(memory_after - memory_before)}")
                optimizer.zero_grad()
                t1 = time.time()
                susage.train()
                if args.restrict_receptive_field:
                    output = susage.forward(torch.FloatTensor(feat_data[input_nodes]).to(device), adjs)
                else:
                    output = susage.forward(feat_data[input_nodes], adjs) # + 9.3477
                if args.sampler == 'full':
                    output = output[output_nodes]
                if multiclass:
                    if args.restrict_receptive_field:
                        loss_train = F.binary_cross_entropy_with_logits(output, torch.FloatTensor(labels[output_nodes]).to(device))
                    else:
                        loss_train = F.binary_cross_entropy_with_logits(output, labels[output_nodes].float())
                else:
                    if args.restrict_receptive_field:
                        loss_train = F.cross_entropy(output, torch.LongTensor(labels[output_nodes]).to(device))
                    else:
                        loss_train = F.cross_entropy(output, labels[output_nodes])
                loss_train.backward() # + 8.4009
                torch.nn.utils.clip_grad_norm_(susage.parameters(), 0.2)
                optimizer.step() # + 2.9932
                times += [time.time() - t1]
                train_losses += [loss_train.detach().tolist()]
                del loss_train
                max_memory_allocated = max(max_memory_allocated, torch.cuda.max_memory_allocated())
        susage.eval()
        adjs, input_nodes, output_nodes = valid_data
        adjs = package_mxl(adjs, device)
        if args.restrict_receptive_field:
            output = susage.forward(torch.FloatTensor(feat_data[input_nodes]).to(device), adjs)
        else:
            output = susage.forward(feat_data[input_nodes], adjs)
        if args.sampler == 'full':
            output = output[output_nodes]
        # if multiclass:
        #     loss_fn = F.binary_cross_entropy_with_logits()
        #     predictions = (output >= 0.5).astype(int).cpu()
        # else:
        #     loss_fn = F.cross_entropy()
        #     predictions = output.argmax(dim=1).cpu()
        if multiclass:
            if args.restrict_receptive_field:
                loss_valid = F.binary_cross_entropy_with_logits(output, torch.FloatTensor(labels[output_nodes]).to(device)).detach().tolist()
                valid_f1 = f1_score((output >= 0.5).cpu().int(), labels[output_nodes], average='samples')
            else:
                loss_valid = F.binary_cross_entropy_with_logits(output, labels[output_nodes].float()).detach().tolist()
                valid_f1 = f1_score((output >= 0.5).cpu().int(), labels[output_nodes].cpu(), average='samples')
        else:
            if args.restrict_receptive_field:
                loss_valid = F.cross_entropy(output, torch.LongTensor(labels[output_nodes]).to(device)).detach().tolist()
                valid_f1 = f1_score(output.argmax(dim=1).cpu(), labels[output_nodes], average='micro')
            else:
                loss_valid = F.cross_entropy(output, labels[output_nodes]).detach().tolist()
                valid_f1 = f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')
        print(("Epoch: %d (%.1fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f") % (epoch, np.sum(times), np.average(train_losses), loss_valid, valid_f1))
        if args.logging:
            f.write(f"{epoch},{np.average(train_losses)},{loss_valid},{valid_f1}\n")
        if valid_f1 > best_val + 1e-2:
            best_val = valid_f1
            torch.save(susage, './save/best_model.pt')
            cnt = 0
        else:
            cnt += 1
        total_iters += 1
        if args.early_stopping and cnt == args.n_stops // args.batch_num:
            break
    best_model = torch.load('./save/best_model.pt')
    best_model.eval()
    test_f1s = []
    
    '''
    If using batch sampling for inference:
    '''
    #     for b in np.arange(len(test_nodes) // args.batch_size):
    #         batch_nodes = test_nodes[b * args.batch_size : (b+1) * args.batch_size]
    #         adjs, input_nodes, output_nodes = sampler(np.random.randint(2**32 - 1), batch_nodes,
    #                                     samp_num_list * 20, len(feat_data), lap_matrix, args.n_layers)
    #         adjs = package_mxl(adjs, device)
    #         output = best_model.forward(feat_data[input_nodes], adjs)[output_nodes]
    #         test_f1 = f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')
    #         test_f1s += [test_f1]
    
    '''
    If using full-batch inference:
    '''
    time_1 = time.time()

    batch_nodes = test_nodes
    adjs, input_nodes, output_nodes = default_sampler(np.random.randint(2**32 - 1), batch_nodes,
                                    samp_num_list * 20, len(feat_data), lap_matrix, args.n_layers)
    adjs = package_mxl(adjs, device)
    if args.restrict_receptive_field:
        output = best_model.forward(torch.FloatTensor(feat_data[input_nodes]).to(device), adjs)[output_nodes]
        test_acc, test_f1, test_sens, test_spec = metrics(output.cpu(), labels[output_nodes])
    else:
        output = best_model.forward(feat_data[input_nodes], adjs)[output_nodes]
        test_acc, test_f1, test_sens, test_spec = metrics(output.cpu(), labels[output_nodes].cpu())
    
    # test_f1s = [f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')]

    if args.logging:
        log_times.append(time_1 - time_0)
        log_total_iters.append(total_iters)
        log_max_memory.append(roundsize(max_memory_allocated))
        log_test_acc.append(test_acc)
        log_test_f1.append(test_f1)
        log_test_sens.append(test_sens)
        log_test_spec.append(test_spec)
    
    print('Iteration: %d, Test F1: %.3f' % (oiter, np.average(test_f1)))

# if args.logging:
#     print("times")
#     print(log_times)
#     print("total_batchess")
#     print(log_total_batches)
#     print("max_memorys")
#     print(log_max_memory)
#     print("test_accs")
#     print(log_test_acc)
#     print("test_f1s")
#     print(log_test_f1)
#     print("test_senss")
#     print(log_test_sens)
#     print("test_specs")
#     print(log_test_spec)

if args.logging:
    f2 = open(f"results/final.csv", "a+")
    f2.write(f"{args.dataset}_{args.sampler}_{args.model}_{args.n_layers}layer_restrict{args.restrict_receptive_field}" + "\n")
    f2.write(f"{mean_and_std(log_times)}, {mean_and_std(log_total_iters)}, {mean_and_std(np.array(log_times) / np.array(log_total_iters), 3)}, "
            f"{mean_and_std(log_max_memory)}, {mean_and_std(log_test_acc, 3)}, {mean_and_std(log_test_f1, 3)}, "
            f"{mean_and_std(log_test_sens, 3)}, {mean_and_std(log_test_spec, 3)}\n")

'''
    Visualize the train-test curve
'''

# dt = pd.DataFrame(all_res, columns=['f1-score', 'batch', 'type'])
# sb.lineplot(data = dt, x='batch', y='f1-score', hue='type')
# plt.legend(loc='lower right')
# plt.show()