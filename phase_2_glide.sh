#!/usr/bin/env bash
set -euo pipefail

# Usage: bash ./phase_2_glide.sh <iteration> <total_threads> <path_project> <project>
# Example: bash ./phase_2_glide.sh 1 120 /mnt/data/dk/work/DeepDocking/projects Manuscript_pytorch_2RH1

iteration="$1"
total_threads="$2"
path_project="$3"
project="$4"

logs="$path_project/$project/logs.txt"
file_path="$(sed -n '1p' "$logs")"
protein="$(sed -n '2p' "$logs")"

: "${SLURM_JOB_NAME:=phase_2}"
python jobid_writer.py -pt "$protein" -fp "$file_path" -n_it "$iteration" -jid "$SLURM_JOB_NAME" -jn "$SLURM_JOB_NAME.txt"

mkdir -p "$file_path/$protein/iteration_$iteration/sdf"
cd "$file_path/$protein/iteration_$iteration/sdf"
shopt -s nullglob

smiles=( ../smile/* )
n_jobs=${#smiles[@]}
if (( n_jobs == 0 )); then
  echo "No SMILES files under $(pwd)/../smile; nothing to do."
  exit 0
fi

per_job_threads=$(( total_threads / n_jobs ))
(( per_job_threads < 1 )) && per_job_threads=1

for f in "${smiles[@]}"; do
  base="$(basename "$f")"
  case "${base%%_*}" in
    train) name="training" ;;
    valid) name="validation" ;;
    test)  name="testing" ;;
    *)     echo "Skipping unrecognized file: $f"; continue ;;
  esac
  #! EDIT THIS ONE LINE IF YOU NEED TO CHANGE LIGPREP FLAGS
  ligprep -ns -i 0 -nt -HOST "localhost:$per_job_threads" -ismi "$f" -osd "${name}_sdf.sdf" &
done

wait