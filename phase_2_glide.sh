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

cd "$file_path/$protein/iteration_$iteration"
mkdir -p sdf
shopt -s nullglob

smiles=(smile/*)
n_jobs=${#smiles[@]}
if (( n_jobs == 0 )); then
  echo "No SMILES files under $(pwd)/smile; nothing to do."
  exit 0
fi

per_job_threads=$(( total_threads / n_jobs ))
(( per_job_threads < 1 )) && per_job_threads=1

make_runner () {
  local name="$1"
  local runner="sdf/${name}_conf.sh"
  cat > "$runner" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
infile="$1"   # relative to iteration dir, e.g. smile/train_*.smi
name="$2"     # training|validation|testing
threads="$3"  # threads for LigPrep

# Run from the sdf directory so LigPrep writes outputs here
cd "$(dirname "$0")"  # now CWD is .../iteration_<n>/sdf

#! edit ligprep command here, just paste in the parameters. 
#! this one is what I want to use for the comparison to my old results
ligprep -ns -i 0 -nt -HOST "localhost:$threads" -ismi "../$infile" -osd "${name}_sdf.sdf"
EOF
  chmod +x "$runner"
  echo "$runner"
}

for f in "${smiles[@]}"; do
  base="$(basename "$f")"
  case "${base%%_*}" in
    train) name="training" ;;
    valid) name="validation" ;;
    test)  name="testing" ;;
    *)     echo "Skipping unrecognized file: $f"; continue ;;
  esac
  runner="$(make_runner "$name")"
  bash "$runner" "$f" "$name" "$per_job_threads" &
done

wait