#!/usr/bin/env python3
import os, pynvml

# --- MIG-aware single-device selection (same policy as phase_4/phase_5 eval) ---
DISALLOWED_NAME_BITS = ["T1000"]
FORBIDDEN_CMD_BITS   = ["gdesmond", "icm64.bin"]
REQUIRE_MIN_FREE_GB  = 0.0

def read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().replace(b"\x00", b" ").strip()
        return raw.decode(errors="ignore").lower()
    except Exception:
        return ""

def name_is_disallowed(name: str) -> bool:
    nl = name.lower()
    return any(bit.lower() in nl for bit in DISALLOWED_NAME_BITS)

def handle_has_forbidden_jobs(handle) -> bool:
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
    except Exception:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        pid = getattr(p, "pid", None)
        if pid is None: continue
        cmd = read_cmdline(pid)
        if any(cmd.endswith(bad) or bad in cmd for bad in FORBIDDEN_CMD_BITS):
            return True
    return False

def select_gpu():
    pynvml.nvmlInit()
    try:
        mig_ok = all(hasattr(pynvml, fn) for fn in [
            "nvmlDeviceGetMigMode",
            "nvmlDeviceGetMaxMigDeviceCount",
            "nvmlDeviceGetMigDeviceHandleByIndex",
        ])
        candidates = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            gpu = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(gpu)
            if isinstance(name, bytes): name = name.decode()
            if name_is_disallowed(name): continue

            mode = 0
            if mig_ok:
                try: mode, _ = pynvml.nvmlDeviceGetMigMode(gpu)
                except Exception: mode = 0

            if mode == 0:
                mem = pynvml.nvmlDeviceGetMemoryInfo(gpu)
                if (mem.free / (1024**3)) >= REQUIRE_MIN_FREE_GB and not handle_has_forbidden_jobs(gpu):
                    candidates.append((mem.free, str(i)))
            else:
                max_migs = 0
                try: max_migs = pynvml.nvmlDeviceGetMaxMigDeviceCount(gpu)
                except Exception: pass
                for mi in range(max_migs):
                    try: mig = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(gpu, mi)
                    except Exception: continue
                    mem = pynvml.nvmlDeviceGetMemoryInfo(mig)
                    if (mem.free / (1024**3)) < REQUIRE_MIN_FREE_GB: continue
                    if handle_has_forbidden_jobs(mig): continue
                    uuid = pynvml.nvmlDeviceGetUUID(mig)
                    if isinstance(uuid, bytes): uuid = uuid.decode()
                    candidates.append((mem.free, uuid))
        if not candidates:
            raise SystemExit("[GPU-SELECT] No allowed GPU/MIG available (predictions).")
        return max(candidates, key=lambda c: c[0])[1]
    finally:
        try: pynvml.nvmlShutdown()
        except Exception: pass

selected = select_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(selected)
print(f"Using CUDA_VISIBLE_DEVICES={selected} for prediction")

import argparse, glob, time, warnings
import numpy as np
import pandas as pd
import builtins as __builtin__

warnings.filterwarnings('ignore')

# For debugging purposes only:
def print(*args, **kwargs):
    __builtin__.print('\t sampling: ', end="")
    return __builtin__.print(*args, **kwargs)

parser = argparse.ArgumentParser()
parser.add_argument('-fn','--fn', required=True)
parser.add_argument('-protein','--protein', required=True)
parser.add_argument('-it','--it', required=True)
parser.add_argument('-file_path','--file_path', required=True)
parser.add_argument('-mdd','--morgan_directory', required=True)
io_args = parser.parse_args()

fname     = io_args.fn
protein   = str(io_args.protein)
it        = int(io_args.it)
SAVE_PATH = io_args.file_path  # contains iteration_<it>/
MDD       = io_args.morgan_directory

# --- torch must be imported after any CUDA env setup done by the caller scripts ---
import torch
from ML.ModelsPytorch import PytorchRefactoredModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for prediction: {device}")

# paths
iter_dir   = os.path.join(SAVE_PATH, f'iteration_{it}')
best_dir   = os.path.join(iter_dir, 'best_models')
pred_dir   = os.path.join(iter_dir, 'morgan_1024_predictions')
os.makedirs(pred_dir, exist_ok=True)

# read thresholds (model_no, thresh, cutoff)
thr_df = pd.read_csv(os.path.join(best_dir, 'thresholds.txt'), header=None, names=['model_no','thresh','cutoff'])
thr_df['model_no'] = thr_df['model_no'].astype(int)

# read hyperparams CSV so we can reconstruct the PyTorch model(s)
hp_csv = os.path.join(iter_dir, 'pytorch_hyperparameter_morgan_with_freq_v3.csv')
hp = pd.read_csv(hp_csv, header=None)
hp.columns = [
    "Model_no","Over_sampling","Batch_size","Learning_rate",
    "N_layers","N_units","dropout","weight","cutoff",
    "ROC_AUC","Pr_0_9","tot_left_0_9_mil","auc_te","pr_te","re_te","tot_left_0_9_mil_te","tot_positives",
]

def get_hparams(model_no: int):
    row = hp[hp.Model_no == model_no].iloc[0]
    n_layers = int(row.N_layers)
    bin_array = n_layers * [0, 1]
    return {
        "num_units": int(row.N_units),
        "dropout_rate": float(row.dropout),
        "bin_array": bin_array,
    }

# load models + thresholds
models, thresholds = [], []
for f in sorted(glob.glob(os.path.join(best_dir, 'model_*_pth.pt'))):
    mn = int(os.path.basename(f).split('_')[1])
    if mn not in set(thr_df.model_no.values):  # sanity
        continue
    hps = get_hparams(mn)
    model = PytorchRefactoredModel.load(f, input_shape=1024, hyperparameters=hps).to(device)
    model.eval()
    models.append(model)
    thresholds.append(float(thr_df[thr_df.model_no == mn].thresh.iloc[0]))

print("Number of models to predict:", len(models))

# prediction loop (1,000,000 rows per chunk)
def predict_shard(shard_filename: str) -> int:
    print("Starting Predictions...")
    t0 = time.time()
    # its gonna blow up with 1M if we have chunked the files as such AND we don't have SLURM. There's no way around it. 
    per_time = 250_000
    n_features = 1024
    z_id = []
    X_set = np.zeros((per_time, n_features), dtype=bool)
    total_passed = 0

    in_path = os.path.join(MDD, shard_filename)
    out_path = os.path.join(pred_dir, shard_filename)

    with open(in_path, 'r') as ref:
        no = 0
        for line in ref:
            tmp = line.rstrip().split(',')
            z_id.append(tmp[0])
            for elem in tmp[1:]:
                X_set[no, int(elem)] = True
            no += 1

            if no == per_time:
                Xb = X_set[:no, :]
                # torch inference
                X_tensor = torch.from_numpy(Xb.astype(np.float32)).to(device)
                with torch.inference_mode():
                    probs_per_model = [torch.sigmoid(m(X_tensor)).squeeze().detach().cpu().numpy() for m in models]

                # write out passing molecules (any model above its threshold)
                with open(out_path, 'a') as w:
                    for j in range(len(probs_per_model[0])):
                        is_pass = 0
                        last_prob = None
                        for i, thr in enumerate(thresholds):
                            p = float(probs_per_model[i][j])
                            last_prob = p  # preserve original behavior: write last model's prob
                            if p > thr:
                                is_pass += 1
                        if is_pass >= 1:
                            total_passed += 1
                            w.write(f"{z_id[j]},{last_prob}\n")

                # reset chunk buffers
                X_set.fill(False)
                z_id.clear()
                no = 0

        # tail
        if no != 0:
            Xb = X_set[:no, :]
            X_tensor = torch.from_numpy(Xb.astype(np.float32)).to(device)
            with torch.inference_mode():
                probs_per_model = [torch.sigmoid(m(X_tensor)).squeeze().detach().cpu().numpy() for m in models]
            with open(out_path, 'a') as w:
                for j in range(len(probs_per_model[0])):
                    is_pass = 0
                    last_prob = None
                    for i, thr in enumerate(thresholds):
                        p = float(probs_per_model[i][j])
                        last_prob = p
                        if p > thr:
                            is_pass += 1
                    if is_pass >= 1:
                        total_passed += 1
                        w.write(f"{z_id[j]},{last_prob}\n")

    print("Prediction time:", time.time() - t0)
    return total_passed

count = predict_shard(fname)
with open(os.path.join(pred_dir, 'passed_file_ct.txt'), 'a') as ref:
    ref.write(f"{fname},{count}\n")
