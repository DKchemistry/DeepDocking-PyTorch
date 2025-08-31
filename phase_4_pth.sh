#!/usr/bin/env bash
# phase_4_pth.sh — write simple jobs (runs fine without SLURM)
# Positional args for my sanity :) 
#   $1  current_iteration
#   $2  t_pos (CPUs)
#   $3  path_project
#   $4  project
#   $5  gpu_partition  (ignored in practice; kept for compatibility)
#   $6  tot_number_iterations
#   $7  percent_first_mols   (e.g., 1 for 1%)
#   $8  percent_last_mols    (e.g., 0.1 for 0.1%, 0.01 for 0.01%)
#   $9  recall_value         (e.g., 0.9)
#   $10 time string          (e.g., 00-15:00)
#   $11 conda_env            (e.g., pth_dd)

set -euo pipefail

# --- args ---
env="${11}"
time_str="${10}"
t_pos="$2"
base_dir="$3"
project="$4"
part_gpu="$5"
total_iters="$6"
pfm="$7"
plm="$8"
rec="$9"

# dummy job name if not set 
# jobid_writer.py reads it to log where you are in the process
# so we need something 
: "${SLURM_JOB_NAME:=phase_4}"
export SLURM_JOB_NAME

# try to activate the env; if this fails, we continue (you should definitely activate)
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "$env" || true
fi

# --- read paths from logs.txt ---
logs="$base_dir/$project/logs.txt"
file_path="$(sed -n '1p' "$logs")"
protein="$(sed -n '2p' "$logs")"
morgan_directory="$(sed -n '4p' "$logs")"
smile_directory="$(sed -n '5p' "$logs")"
sof="$(sed -n '6p' "$logs")"      # docking software: Glide or FRED
nhp="$(sed -n '7p' "$logs")"      # number of hyperparameters
num_molec="$(sed -n '8p' "$logs")"

echo "writing jobs"
python jobid_writer.py -pt "$protein" -fp "$file_path" -n_it "$1" -jid "$SLURM_JOB_NAME" -jn "$SLURM_JOB_NAME.txt"

echo "Extracting labels"
if [[ "$sof" == "Glide" ]]; then
  kw='r_i_docking_score'
elif [[ "$sof" == "FRED" ]]; then
  kw='FRED Chemgauss4 score'
else
  echo "Unknown docking software in logs.txt line 6: '$sof'"; exit 1
fi

python scripts_2/extract_labels.py -n_it "$1" -pt "$protein" -fp "$file_path" -t_pos "$t_pos" -score "$kw"

echo "Creating simple jobs"
if [[ "$6" == "$1" ]]; then
  last='True'
else
  last='False'
fi

python scripts_2/simple_job_models_pth.py \
  -n_it "$1" \
  -mdd "$morgan_directory" \
  -time "$time_str" \
  -file_path "$file_path/$protein" \
  -nhp "$nhp" \
  -titr "$total_iters" \
  -n_mol "$num_molec" \
  -pfm "$pfm" \
  -plm "$plm" \
  -ct "$rec" \
  -gp "$part_gpu" \
  -tf_e "$env" \
  -isl "$last"

# python makes these files and we make a dir variable
sj_dir="$file_path/$protein/iteration_${1}/simple_job"

#say that we are launching unless there is simple job dir, very unliklely but fine
echo "Launching simple jobs from: $sj_dir"
if [[ ! -d "$sj_dir" ]]; then
  echo "ERROR: directory not found: $sj_dir"; exit 1
fi

# if a glob doesn’t match anything, expand to nothing instead of the literal pattern
# and I mean literal, jobs (bash array) would contain these characters: "simple_job_*.sh"
# this global so we will turn it off
shopt -s nullglob
# makes a bash array (like a python list)
jobs=( "$sj_dir"/simple_job_*.sh )
shopt -u nullglob

# check if we have the scripts
if (( ${#jobs[@]} == 0 )); then
  echo "No simple_job_*.sh scripts found in $sj_dir"; exit 1
fi

# make executable
chmod +x "${jobs[@]}"

# fire the jobs with logged output, sleep one second in between (later we may want something smarter but right now this is fine)
for j in "${jobs[@]}"; do
  echo "Starting $j"
  "$j" > "${j}.log" 2>&1 &
  sleep 1
done

wait
echo "All jobs finished (or crashed) check how many you expect to have been added to pytorch_hyperparameter_morgan_with_freq_v3.csv based on the job set up for a quick sanity check."