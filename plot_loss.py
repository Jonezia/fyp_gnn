#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%
def custom_aggregation(group):
    return group.sum() / len(group)

#%%
def generate_average_run(start_directory, oiter=5):
    # epoch,train_loss,val_loss,val_f1
    total_runs = pd.DataFrame()
    for i in range(oiter):
        run = pd.read_csv(start_directory + f"_{i}.csv", index_col=False)
        total_runs = pd.concat([total_runs, run])
    avg_runs = total_runs.groupby(total_runs.epoch).apply(custom_aggregation)
    return avg_runs

# pubmed_ladies_GCN_5layer_restrictFalse_avg_runs
GCN_runs = generate_average_run("results/per_epoch/reddit2_ladies_GCN_2layer_fullbatch", oiter=1)
# scalarGCN_runs = generate_average_run("results/per_epoch/flickr_ladies_scalarGCN_5layer_fullbatch")

#%%
plt.plot(GCN_runs.epoch, GCN_runs.train_loss, label="GCN")
# plt.plot(scalarGCN_runs.epoch, scalarGCN_runs.train_loss, label="ScalarGCN")
plt.legend(loc="upper right", fontsize=8)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss Curves")
plt.show()

#%%
plt.plot(GCN_runs.epoch, GCN_runs.val_loss, label="GCN")
# plt.plot(scalarGCN_runs.epoch, scalarGCN_runs.val_loss, label="ScalarGCN")
plt.legend(loc="upper right", fontsize=8)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss Curves")
plt.show()

#%%
plt.plot(GCN_runs.epoch, GCN_runs.val_f1, label="GCN")
# plt.plot(scalarGCN_runs.epoch, scalarGCN_runs.val_f1, label="ScalarGCN")
plt.legend(loc="lower right", fontsize=8)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation F1 Curves")
plt.show()

# %%
