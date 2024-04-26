"""
V 2.2.1
"""

import gc
import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve, roc_curve

START_TIME = time.time()
print("Parsing args...")
parser = argparse.ArgumentParser()
parser.add_argument("-num_units", "--nu", required=True)
parser.add_argument("-dropout", "--df", required=True)
parser.add_argument("-learn_rate", "--lr", required=True)
parser.add_argument("-bin_array", "--ba", required=True)
parser.add_argument("-wt", "--wt", required=True)
parser.add_argument("-cf", "--cf", required=True)
parser.add_argument("-rec", "--rec", required=True)
parser.add_argument("-n_it", "--n_it", required=True)
parser.add_argument("-t_mol", "--t_mol", required=True)
parser.add_argument("-bs", "--bs", required=True)
parser.add_argument("-os", "--os", required=True)
parser.add_argument("-d_path", "--data_path", required=True)  # new!

# adding parameter for where to save all the data to:
parser.add_argument("-s_path", "--save_path", required=False, default=None)

# allowing for variable number of molecules to validate and test from:
parser.add_argument("-n_mol", "--number_mol", required=False, default=1000000)
parser.add_argument("-t_n_mol", "--train_num_mol", required=False, default=-1)
parser.add_argument(
    "-cont", "--continuous", required=False, action="store_true"
)  # Using binary or continuous labels
parser.add_argument(
    "-smile", "--smiles", required=False, action="store_true"
)  # Using smiles or morgan as or continuous labels
parser.add_argument(
    "-norm", "--normalization", required=False, action="store_false"
)  # if continuous labels are used -> normalize them?

io_args = parser.parse_args()

print(sys.argv)

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
        #smiles_path.format("train"),
        morgan_path.format("train"),
        labels_path.format("training"),
    )
    test_data = get_data(
        #smiles_path.format("test"),
        morgan_path.format("test"),
        labels_path.format("testing"),
    )
    valid_data = get_data(
       #smiles_path.format("valid"),
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

# This is our new model
hyperparameters = {
    "bin_array": ba * [0, 1],
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

# save training data

# Construct the path for saving
nd_arrays_path = os.path.join(SAVE_PATH, f"iteration_{n_iteration}", "nd_arrays")

# Ensure the directory exists
os.makedirs(nd_arrays_path, exist_ok=True)

# consistent file name 
x_train_filename = "Oversampled_X_train_iteration.npy"
y_train_filename = "Oversampled_y_train_iteration.npy"

# Full paths
x_train_filepath = os.path.join(nd_arrays_path, x_train_filename)
y_train_filepath = os.path.join(nd_arrays_path, y_train_filename)

# Save the arrays
np.save(x_train_filepath, Oversampled_X_train)
np.save(y_train_filepath, Oversampled_y_train)

print(f"Saved Oversampled_X_train to {x_train_filepath}")
print(f"Saved Oversampled_y_train to {y_train_filepath}")
