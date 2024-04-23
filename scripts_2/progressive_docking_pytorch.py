import gc
import argparse
import glob
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from  ML.ModelsPytorch import PytorchRefactoredModel

# Time tracking and argument parsing
START_TIME = time.time()
print("Parsing args...")
parser = argparse.ArgumentParser()
parser.add_argument("-num_units", "--nu", type=int, required=True)
parser.add_argument("-dropout", "--df", type=float, required=True)
parser.add_argument("-learn_rate", "--lr", type=float, required=True)
parser.add_argument(
    "-bin_array", "--ba", type=int, required=True
)  # This might still need adjustment based on actual bin_array handling? Hard to tell
   # if this is because of how tensorflow handles layers where PyTorch might use squeeze?
parser.add_argument("-wt", "--wt", type=float, required=True)
parser.add_argument("-cf", "--cf", type=float, required=True)
parser.add_argument("-rec", "--rec", type=float, required=True)
parser.add_argument("-n_it", "--n_it", type=int, required=True)
parser.add_argument("-t_mol", "--t_mol", type=float, required=True)
parser.add_argument("-bs", "--bs", type=int, required=True)
parser.add_argument("-os", "--os", type=int, required=True)
parser.add_argument("-d_path", "--data_path", required=True)
parser.add_argument("-s_path", "--save_path", default=None)
parser.add_argument("-n_mol", "--number_mol", type=int, default=1000000)
parser.add_argument("-t_n_mol", "--train_num_mol", type=int, default=-1)


io_args = parser.parse_args()

nu = int(io_args.nu)
df = float(io_args.df)
lr = float(io_args.lr)
ba = int(io_args.ba)
wt = float(io_args.wt)
cf = float(io_args.cf)
rec = float(io_args.rec)
n_it = int(io_args.n_it)
bs = int(io_args.bs)
oss = int(io_args.os)
t_mol = float(io_args.t_mol)

CONTINUOUS = io_args.continuous # is this used?
NORMALIZE = io_args.normalization # is this used? 
#SMILES = io_args.smiles #not used 
TRAINING_SIZE = int(io_args.train_num_mol)
num_molec = int(io_args.number_mol)

DATA_PATH = io_args.data_path  # Now == file_path/protein
SAVE_PATH = io_args.save_path
# if no save path is provided we just save it in the same location as the data
if SAVE_PATH is None:
    SAVE_PATH = DATA_PATH


print(nu, df, lr, ba, wt, cf, bs, oss, DATA_PATH)
if TRAINING_SIZE == -1:
    print("Training size not specified, using entire dataset...")
print("Finished parsing args...")

#! Still need to get the data, might be as simple as .from_numpy?

# Simple hyperparameters setup based on argparse inputs
hyperparameters = {
    "bin_array": ba * [0, 1], # I am still so confused what this is doing? 
    "dropout_rate": df,
    "learning_rate": lr,
    "num_units": nu,
    "batch_size": bs,
    "class_weight": wt,
    "epsilon": 1e-06,
}

# Assuming data loading and preprocessing is handled elsewhere
# Dummy setup for illustration:
features = np.random.randn(io_args.number_mol, 1024)
labels = np.random.randint(0, 2, size=(io_args.number_mol, 1))
dataset = TensorDataset(
    torch.tensor(features, dtype=torch.float32),
    torch.tensor(labels, dtype=torch.float32),
)
dataloader = DataLoader(dataset, batch_size=io_args.bs, shuffle=True)

# Model initialization
model = PytorchRefactoredModel(
    input_shape=1024, hyperparameters=hyperparameters
)  
optimizer = optim.Adam(model.parameters(), lr=io_args.lr)
loss_function = nn.BCEWithLogitsLoss()


# Training loop (does it really need to be a function? Safer I guess)
def train_model(model, dataloader, optimizer, loss_function):
    model.train()
    for epoch in range(io_args.n_it):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


train_model(model, dataloader, optimizer, loss_function)

print("Training completed in:", time.time() - START_TIME, "seconds")

