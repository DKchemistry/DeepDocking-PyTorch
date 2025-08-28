#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --job-name=phase_3
set -euo pipefail

# Usage: bash ./phase_3_glide.sh <iteration> <total_threads> <path_project> <project>
# Example: bash ./phase_3_glide.sh 1 150 /mnt/data/dk/work/DeepDocking/projects Manuscript_pytorch_2RH1

iteration="$1"
total_threads="$2"      # desired TOTAL across the 3 parallel Glide runs, so 150 in that example 
path_project="$3"
project="$4"

# read paths from logs.txt
logs="$path_project/$project/logs.txt"
file_path="$(sed -n '1p' "$logs")"
protein="$(sed -n '2p' "$logs")"
grid_file="$(sed -n '3p' "$logs")"
glide_input_file="$(sed -n '9p' "$logs")"

# keep jobid_writer behavior even without Slurm
: "${SLURM_JOB_NAME:=phase_3}"
python jobid_writer.py -pt "$protein" -fp "$file_path" -n_it "$iteration" -jid "$SLURM_JOB_NAME" -jn "$SLURM_JOB_NAME.txt"

# generate Glide .in files for this iteration
python scripts_1/input_glide.py -pt "$protein" -fp "$file_path" -gf "$grid_file" -n_it "$iteration" -g_in "$glide_input_file"

# run from the docked directory so Glide writes outputs here
cd "$file_path/$protein/iteration_$iteration/docked"
shopt -s nullglob
ins=( *.in )
n_in=${#ins[@]}
if (( n_in == 0 )); then
  echo "No .in files found in $(pwd)"
  exit 0
fi

# Sorry this is written confusingly, because we always are going to have
# 3 Glide jobs (testing, training, validation)
# We split total CPU threads evenly: each job gets localhost:(total_threads/3) 
# So, we set max CPUs (e.g. 150 in example) for localhost:N
# Then, we get localhost:50, -njobs 150
# This is a pretty typical situation for our servers
# But you can scale that up or down
per_job_threads=$(( total_threads / n_in ))
(( per_job_threads < 1 )) && per_job_threads=1
njobs="$total_threads"

# launch each Glide job in parallel
for f in "${ins[@]}"; do
  jobname="phase_3_${f%.*}"
  "$SCHRODINGER/glide" -HOST "localhost:${per_job_threads}" -NJOBS "$njobs" -OVERWRITE -JOBNAME "$jobname" "$f" &
done

wait
