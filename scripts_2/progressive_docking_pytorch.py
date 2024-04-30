import glob
import gc
import argparse
import pynvml
import os
import random
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc

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
from torch.utils.tensorboard import SummaryWriter 

from ML.ModelsPytorch import PytorchRefactoredModel
from ML.DDPytorchCallbacks import EarlyStopping, TimedStopping

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
total_mols = t_mol

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


def get_oversampled_morgan(Oversampled_zid, fname):
    print("x data from:", fname)
    # Gets only the morgan fingerprints of those randomly selected zinc ids
    with open(fname, "r") as ref:
        for line in ref:
            tmp = line.rstrip().split(",")

            # only extracting those that were randomly selected
            if (tmp[0] in Oversampled_zid.keys()) and (
                type(Oversampled_zid[tmp[0]]) != np.ndarray
            ):
                train_set = np.zeros([1, 1024], dtype=np.bool_)
                on_bit_vector = tmp[1:]

                for elem in on_bit_vector:
                    train_set[0, int(elem)] = 1

                # creates a n x 1024 numpy ndarray where n is the number of times that zinc id was randomly selected
                Oversampled_zid[tmp[0]] = np.repeat(
                    train_set, Oversampled_zid[tmp[0]], axis=0
                )
    return Oversampled_zid


def get_morgan_and_scores(morgan_path, ID_labels):
    # ID_labels is a dataframe containing the zincIDs and their corresponding scores.
    train_set = np.zeros([num_molec, 1024], dtype=bool)  # using bool to save space
    train_id = []
    print("x data from:", morgan_path)
    with open(morgan_path, "r") as ref:
        line_no = 0
        for line in ref:
            if line_no >= num_molec:
                break

            mol_info = line.rstrip().split(",")
            train_id.append(mol_info[0])

            # "Decompressing" the information from the file about where the 1s are on the 1024 bit vector.
            bit_indicies = mol_info[
                1:
            ]  # array of indexes of the binary 1s in the 1024 bit vector representing the morgan fingerprint
            for elem in bit_indicies:
                train_set[line_no, int(elem)] = 1

            line_no += 1

    train_set = train_set[:line_no, :]

    print("Done...")
    train_pd = pd.DataFrame(data=train_set, dtype=np.bool_)
    train_pd["ZINC_ID"] = train_id

    ID_labels = ID_labels.to_frame()
    print(ID_labels.columns)
    score_col = ID_labels.columns.difference(["ZINC_ID"])[0]
    print(score_col)

    train_data = pd.merge(ID_labels, train_pd, how="inner", on=["ZINC_ID"])
    X_train = train_data[
        train_data.columns.difference(["ZINC_ID", score_col])
    ].values  # input
    y_train = train_data[[score_col]].values  # labels
    return X_train, y_train


# Gets the labels data
def get_data(morgan_path, labels_path):
    # Load the docking scores (with corresponding Zinc_IDs)
    labels = pd.read_csv(labels_path, sep=",", header=0)

    # Load Morgan fingerprints, only reading in the Zinc IDs
    morgan = pd.read_csv(morgan_path, usecols=[0], header=None, names=["ZINC_ID"])

    # Merge the Morgan data with the labels data on 'ZINC_ID'
    data = morgan.merge(labels, on="ZINC_ID")

    # Set the index of the dataframe to 'ZINC_ID'
    data.set_index("ZINC_ID", inplace=True)

    return data


n_iteration = n_it
total_mols = t_mol

try:
    os.mkdir(SAVE_PATH + "/iteration_" + str(n_iteration) + "/all_models")
except OSError:
    pass

try:
    os.mkdir(SAVE_PATH + "/iteration_" + str(n_iteration) + "/nd_arrays")
except OSError:
    pass

# Getting data from prev iterations and this iteration
data_from_prev = pd.DataFrame()
train_data = pd.DataFrame()
test_data = pd.DataFrame()
valid_data = pd.DataFrame()
y_valid_first = pd.DataFrame()
y_test_first = pd.DataFrame()
for i in range(1, n_iteration + 1):

    # getting all the data
    print("\nGetting data from iteration", i)
    smiles_path = (
        DATA_PATH + "/iteration_" + str(i) + "/smile/{}_smiles_final_updated.smi"
    )
    morgan_path = (
        DATA_PATH + "/iteration_" + str(i) + "/morgan/{}_morgan_1024_updated.csv"
    )
    labels_path = DATA_PATH + "/iteration_" + str(i) + "/{}_labels.txt"

    # Resulting dataframe will have cols of smiles (if selected) and docking scores with an index of Zinc IDs
    train_data = get_data(
        # smiles_path.format("train"),
        morgan_path.format("train"),
        labels_path.format("training"),
    )
    test_data = get_data(
        # smiles_path.format("test"),
        morgan_path.format("test"),
        labels_path.format("testing"),
    )
    valid_data = get_data(
        # smiles_path.format("valid"),
        morgan_path.format("valid"),
        labels_path.format("validation"),
    )

    print("Data acquired...")
    print(
        "Train shape:",
        train_data.shape,
        "Valid shape:",
        valid_data.shape,
        "Test shape:",
        test_data.shape,
    )

    if i == 1:  # for the first iteration we only add the training data
        # because test and valid from this iteration is used by all subsequent iterations (constant dataset).

        y_test_first = (
            test_data  # test and valid should be seperate from training dataset
        )
        y_valid_first = valid_data
        y_old = train_data
    elif i == n_iteration:
        break
    else:
        y_old = pd.concat([train_data, valid_data, test_data], axis=0)

    data_from_prev = pd.concat([y_old, data_from_prev], axis=0)

    print(
        "Data Augmentation iteration {} data shape: {}".format(i, data_from_prev.shape)
    )

# Always using the same valid and test dataset across all iterations:
if n_iteration != 1:
    train_data = pd.concat(
        [train_data, test_data, valid_data], axis=0
    )  # These datasets are from the current iteration.
    train_data = pd.concat(
        [train_data, data_from_prev]
    )  # combining all the datasets into a single training set for iterations after the first
elif (n_iteration == 1) and (TRAINING_SIZE != -1):
    if TRAINING_SIZE > len(
        train_data
    ):  # If a training set size larger than all the available training data (number of lines in training_set_labels.txt excluding the first) is chosen, use all the available data and print a warning
        print("Maximum training size reached. Using all training data...")
    else:  # If the size is less than all the available training data, randomly sample the dataset
        train_data = train_data.sample(n=TRAINING_SIZE)

print("Training labels shape: ", train_data.shape)

# Exiting if there are not enough hits
if (y_valid_first.r_i_docking_score < cf).values.sum() <= 10 or (
    y_test_first.r_i_docking_score < cf
).values.sum() <= 10:
    print("There are not enough hits... exiting.")
    sys.exit()


print("Using binary labels...")
# valid and testing data is from the first iteration.
y_valid = y_valid_first.r_i_docking_score < cf
y_test = y_test_first.r_i_docking_score < cf
y_train = train_data.r_i_docking_score < cf

# Getting all the ids of hits and non hits.
y_pos = y_train[y_train == 1]  # true
y_neg = y_train[y_train == 0]  # false

print("Converting y_pos and y_neg to dict (for faster access time)")
y_pos = y_pos.to_dict()
y_neg = y_neg.to_dict()

num_neg = len(y_neg)
num_pos = len(y_pos)

sample_size = np.min([num_neg, num_pos * oss])

print("\nOversampling...", "size:", sample_size)
print("\tNum pos: {} \n\tNum neg: {}".format(num_pos, num_neg))
Oversampled_zid = {}  # Keeps track of how many times that zinc_id is randomly selected
Oversampled_zid_y = {}

pos_keys = list(y_pos.keys())
neg_keys = list(y_neg.keys())

for i in range(sample_size):
    # Randomly sampling equal number of hits and misses:
    idx = random.randint(0, num_pos - 1)
    idx_neg = random.randint(0, num_neg - 1)
    pos_zid = pos_keys[idx]
    neg_zid = neg_keys[idx_neg]

    # Adding both pos and neg to the dictionary
    try:
        Oversampled_zid[pos_zid] += 1
    except KeyError:
        Oversampled_zid[pos_zid] = 1
        Oversampled_zid_y[pos_zid] = y_pos[pos_zid]

    try:
        Oversampled_zid[neg_zid] += 1
    except KeyError:
        Oversampled_zid[neg_zid] = 1
        Oversampled_zid_y[neg_zid] = y_neg[neg_zid]

Oversampled_X_train = np.zeros([sample_size * 2, 1024], dtype=np.bool_)
print("Using morgan fingerprints...")
# this part is what gets the morgan fingerprints:
print(
    "looking through file path:",
    DATA_PATH + "/iteration_" + str(n_iteration) + "/morgan/*",
)

for i in range(1, n_iteration + 1):
    for f in glob.glob(DATA_PATH + "/iteration_" + str(i) + "/morgan/*"):
        set_name = f.split("/")[-1].split("_")[0]
        print("\t", set_name)
        # Valid and test datasets are always going to be from the first iteration.
        if i == 1:
            if set_name == "valid":
                X_valid, y_valid = get_morgan_and_scores(f, y_valid)
            elif set_name == "test":
                X_test, y_test = get_morgan_and_scores(f, y_test)

        # Fills the dictionary with the actual morgan fingerprints
        Oversampled_zid = get_oversampled_morgan(Oversampled_zid, f)

print("y validation shape:", y_valid.shape)

ct = 0
Oversampled_y_train = np.zeros([sample_size * 2, 1])
print("oversampled sample:", list(Oversampled_zid.items())[0])
num_morgan_missing = 0
for key in Oversampled_zid.keys():
    try:
        tt = len(Oversampled_zid[key])
    except TypeError as e:
        # print("Missing morgan fingerprint for this ZINC ID")
        # print(key, Oversampled_zid[key])
        num_morgan_missing += 1
        continue  # Skipping data that has no labels for it

    Oversampled_X_train[ct : ct + tt] = Oversampled_zid[
        key
    ]  # repeating the same data for as many times as it was selected
    Oversampled_y_train[ct : ct + tt] = Oversampled_zid_y[key]
    ct += tt

print("Done oversampling, number of missing morgan fingerprints:", num_morgan_missing)

# FREE MEMORY

del data_from_prev
del y_neg
del neg_keys
del y_pos
del pos_keys
del test_data
del valid_data
del Oversampled_zid
del Oversampled_zid_y
del y_valid_first
del y_test_first
del y_train
del y_old

gc.collect()

# END FREE MEMORY

print("Data prep time:", time.time() - START_TIME)
print("Configuring model...")
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

print("\n" + "-" * 20)
print("Training data info:" + "\n")
print("X Data Shape[1:]", Oversampled_X_train.shape[1:])
print("X Data Shape", Oversampled_X_train.shape)
print("X Data example", Oversampled_X_train[0])
print("Hyperparameters", hyperparameters)

# Assuming Oversampled_X_train and Oversampled_y_train are already filled and have the correct shapes
# Convert to tensors
X_train_tensor = torch.tensor(Oversampled_X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(Oversampled_y_train, dtype=torch.float32).reshape(
    -1
)  # Ensuring y_train is a 1D array

# Create TensorDataset and DataLoader for training
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(dataset, batch_size=io_args.bs, shuffle=True)

# Print to confirm
print("Training data tensor shapes:", X_train_tensor.shape, y_train_tensor.shape)
print("DataLoader and TensorDataset prepared for training.")

# Convert to tensors
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).reshape(
    -1
)  # Ensuring y_train is a 1D array
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(
    -1
)  # Ensuring y_train is a 1D array

# Create TensorDatasets and DataLoaders
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
valid_dataloader = DataLoader(valid_dataset, batch_size=io_args.bs, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=io_args.bs, shuffle=False)

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
    save_path,
    iteration,
    model_number,
):
    writer = get_tensorboard_writer(save_path, iteration, model_number)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_save_path)
    timed_stopping = TimedStopping(max_seconds=3600)

    best_validation_loss = np.Inf
    # epochs matching tf version 
    for epoch in range(500):
        model.train()
        # batch_idx is for fine grained tracking in tensorboard
        # might not keep it
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

            # Log training loss per batch
            writer.add_scalar(
                "Loss/Train", loss.item(), epoch * len(train_dataloader) + batch_idx
            )

        val_loss = validate(model, valid_dataloader, loss_function, device)
        writer.add_scalar("Loss/Validate", val_loss, epoch)

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        if timed_stopping.should_stop():
            print("Timed stopping triggered")
            break

    if best_validation_loss == early_stopping.val_loss_min:
        print("Best model saved during training")
    else:
        model.save(
            model_save_path
        )  # Save at end if the best was not during early stopping
        print(f"Model saved after training to {model_save_path}")

    writer.close()


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


def get_tensorboard_writer(save_path, iteration, model_number):
    # Create a directory for TensorBoard logs inside the iteration directory
    tb_log_dir = os.path.join(
        save_path, f"iteration_{iteration}", "tensorboard_logs", f"model_{model_number}"
    )
    os.makedirs(tb_log_dir, exist_ok=True)
    return SummaryWriter(log_dir=tb_log_dir)

print("Starting training...")

train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    loss_function,
    device,
    model_save_path,
    SAVE_PATH,  # TensorBoard save path
    n_it,
    mn,
)

END_TIME = time.time() - START_TIME
print("Training completed in:", END_TIME, "seconds")

# Prediction onto testing and validation


def generate_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probabilities = torch.sigmoid(
                logits
            ).squeeze()  # Convert logits to probabilities
            predictions.extend(probabilities.cpu().numpy())  # Collect predictions
    return np.array(predictions)

print("Generating predictions on validation...")
prediction_valid = generate_predictions(model, valid_dataloader, device)
print("Generating predictions on testing...")
prediction_test = generate_predictions(model, test_dataloader, device)


print("Getting stats from predictions...")
# Getting stats for validation
precision_vl, recall_vl, thresholds_vl = precision_recall_curve(
    y_valid, prediction_valid
)
fpr_vl, tpr_vl, thresh_vl = roc_curve(y_valid, prediction_valid)
auc_vl = auc(fpr_vl, tpr_vl)
pr_vl = precision_vl[np.where(recall_vl > rec)[0][-1]]
pos_ct_orig = np.sum(y_valid)
Total_left = rec * pos_ct_orig / pr_vl * total_mols * 1000000 / len(y_valid)
tr = thresholds_vl[np.where(recall_vl > rec)[0][-1]]

# Getting stats for testing
precision_te, recall_te, thresholds_te = precision_recall_curve(y_test, prediction_test)
fpr_te, tpr_te, thresh_te = roc_curve(y_test, prediction_test)
auc_te = auc(fpr_te, tpr_te)
pr_te = precision_te[np.where(thresholds_te > tr)[0][0]]
re_te = recall_te[np.where(thresholds_te > tr)[0][0]]
pos_ct_orig = np.sum(y_test)
Total_left_te = re_te * pos_ct_orig / pr_te * total_mols * 1000000 / len(y_test)
print("Stats collected.")

with open(
    SAVE_PATH
    + "/iteration_"
    + str(n_it)
    + "/pytorch_hyperparameter_morgan_with_freq_v3.csv",
    "a",
) as ref:
    ref.write(
        str(mn)
        + ","
        + str(oss)
        + ","
        + str(bs)
        + ","
        + str(lr)
        + ","
        + str(ba)
        + ","
        + str(nu)
        + ","
        + str(df)
        + ","
        + str(wt)
        + ","
        + str(cf)
        + ","
        + str(auc_vl)
        + ","
        + str(pr_vl)
        + ","
        + str(Total_left)
        + ","
        + str(auc_te)
        + ","
        + str(pr_te)
        + ","
        + str(re_te)
        + ","
        + str(Total_left_te)
        + ","
        + str(pos_ct_orig)
        + "\n"
    )

with open(
    SAVE_PATH
    + "/iteration_"
    + str(n_it)
    + "/pytorch_hyperparameter_morgan_with_freq_v3.txt",
    "a",
) as ref:
    # The sting of hyperparameters that stores what will be appended to the file ref
    hp = "\n" + "-" * 15 + "\n" + "Hyperparameters:" + "\n"
    hp += "- Model Number: " + str(mn) + "\n"
    hp += "- Training Time: " + str(round(END_TIME, 3)) + "\n"
    hp += "  - OS: " + str(oss) + "\n"
    hp += "  - Batch Size: " + str(bs) + "\n"
    hp += "  - Learning Rate: " + str(lr) + "\n"
    hp += "  - Bin Array: " + str(ba) + "\n"
    hp += "  - Num. Units: " + str(nu) + "\n"
    hp += "  - Dropout Freq.: " + str(df) + "\n" * 2

    hp += "  - Class Weight Parameter wt: " + str(wt) + "\n"
    hp += "  - cf: " + str(cf) + "\n"
    hp += "  - auc vl: " + str(auc_vl) + "\n"
    hp += "  - auc te: " + str(auc_te) + "\n"
    hp += "  - Precision validation: " + str(pr_vl) + "\n"
    hp += "  - Precision testing: " + str(pr_te) + "\n"
    hp += "  - Recall testing: " + str(re_te) + "\n"
    hp += "  - Pos ct orig: " + str(pos_ct_orig) + "\n"
    hp += "  - Total Left: " + str(Total_left) + "\n"
    hp += "  - Total Left testing: " + str(Total_left_te) + "\n\n"

    hp += "-" * 15
    ref.write(hp)

print("Model number", mn, "complete.")
