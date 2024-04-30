import argparse
import pynvml
import os
import random
import sys
import time
import numpy as np
import pandas as pd

pynvml.nvmlInit()

def select_gpu():
    device_count = pynvml.nvmlDeviceGetCount()
    min_memory = float("inf")
    selected_device = None
    free_gpu_found = False

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        
        if len(procs) == 0:
            print(f"GPU {i} is free of any compute processes. Selecting this GPU.")
            selected_device = i
            free_gpu_found = True
            break
        elif mem_info.free < min_memory:
            selected_device = i
            min_memory = mem_info.free

    if not free_gpu_found:
        print(f"No completely free GPUs found. Selecting GPU {selected_device} with {min_memory / (1024**3):.2f} GB free memory.")
    else:
        print(f"Selected GPU {selected_device} as it is free of compute processes.")

    pynvml.nvmlShutdown()
    return selected_device

selected_gpu = select_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)

# Torch imports have to come after os.environ call.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split


from ML.ModelsPytorch import PytorchRefactoredModel
from ML.DDPytorchCallbacks import EarlyStopping

# Time tracking and argument parsing
START_TIME = time.time()

# GPU device usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Parsing args...")
parser = argparse.ArgumentParser()
parser.add_argument("-num_units", "--nu", type=int, required=True)
parser.add_argument("-dropout", "--df", type=float, required=True)
parser.add_argument("-learn_rate", "--lr", type=float, required=True)
parser.add_argument(
    "-bin_array", "--ba", type=int, required=True
)  # handles layer architecture by interlacing dropout
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


# Simple hyperparameters setup based on argparse inputs
hyperparameters = {
    "bin_array": ba * [0, 1],
    # if 0 = nn.Linear -> nn.BatchNorm1d -> nn.ReLU
    # if 1 = nn.Dropout(dropout_rate)
    "dropout_rate": df,
    "learning_rate": lr,
    "num_units": nu,
    "batch_size": bs,
    "class_weight": wt,
    "epsilon": 1e-06,
}

# This approach will have to be changed as oversampling is done on data collection
print("Loading preprocessed data...")

# Load preprocessed data
# Load validation and test data, always from iteration_1
x_valid_path = os.path.join(SAVE_PATH, "iteration_1", "nd_arrays", "X_valid.npy")
y_valid_path = os.path.join(SAVE_PATH, "iteration_1", "nd_arrays", "y_valid.npy")
x_test_path = os.path.join(SAVE_PATH, "iteration_1", "nd_arrays", "X_test.npy")
y_test_path = os.path.join(SAVE_PATH, "iteration_1", "nd_arrays", "y_test.npy")

X_valid = np.load(x_valid_path)
y_valid = np.load(y_valid_path).reshape(-1)
X_test = np.load(x_test_path)
y_test = np.load(y_test_path).reshape(-1)

# Convert to tensors
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create TensorDatasets and DataLoaders
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
valid_dataloader = DataLoader(valid_dataset, batch_size=io_args.bs, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=io_args.bs, shuffle=False)

# Load training data, the training data is calling the current directory, BUT
# the data script is getting training data from all iterations current and prior
x_train_path = os.path.join(
    SAVE_PATH, f"iteration_{n_it}/nd_arrays/Oversampled_X_train_iteration.npy"
)
y_train_path = os.path.join(
    SAVE_PATH, f"iteration_{n_it}/nd_arrays/Oversampled_y_train_iteration.npy"
)

X_train = np.load(x_train_path)
y_train = np.load(y_train_path).reshape(-1)  # Reshape y_train to ensure it's a 1D array

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Create TensorDataset and DataLoader
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(dataset, batch_size=io_args.bs, shuffle=True)

print("Data loaded and DataLoader created.")

print("Init model...")
# Model initialization
model = PytorchRefactoredModel(input_shape=1024, hyperparameters=hyperparameters).to(
    device
)
optimizer = optim.Adam(model.parameters(), lr=io_args.lr)
# Inversely weight class 1 due to TF -> PyTorch differences
inverse_wt = 1.0 / wt  
loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(inverse_wt).to(device))


def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    loss_function,
    device,
    model_save_path,
):
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_save_path)

    best_validation_loss = np.Inf
    for epoch in range(10):
        model.train()
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

        val_loss = validate(model, valid_dataloader, loss_function, device)
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    if best_validation_loss == early_stopping.val_loss_min:
        print("Best model saved during training")
    else:
        model.save(
            model_save_path
        )  # Save at end if the best was not during early stopping
        print(f"Model saved after training to {model_save_path}")


def validate(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0
    # using .inference_mode() instead of .no_grad()
    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets.unsqueeze(1))
            total_loss += loss.item()
    model.train()
    return total_loss / len(dataloader)


def manage_model_number(save_path, iteration):
    """Manages the incrementation of model number stored in a text file."""
    model_no_path = os.path.join(save_path, f"iteration_{iteration}", "model_no.txt")
    try:
        with open(model_no_path, "r") as file:
            model_number = int(file.read().strip()) + 1
    except FileNotFoundError:
        model_number = 1

    with open(model_no_path, "w") as file:
        file.write(str(model_number))

    return model_number

mn = manage_model_number(SAVE_PATH, n_it)

model_save_path = os.path.join(
    SAVE_PATH, f"iteration_{n_it}", "all_models", f"model_{mn}_pth.pt"
)

print("Starting training...")

train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    loss_function,
    device,
    model_save_path,
)

print("Training completed in:", time.time() - START_TIME, "seconds")
