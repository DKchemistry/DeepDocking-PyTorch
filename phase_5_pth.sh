#!/usr/bin/env bash
set -euo pipefail

# dummy job name if not set 
# jobid_writer.py reads it to log where you are in the process
# so we need something 
: "${SLURM_JOB_NAME:=phase_5}"
export SLURM_JOB_NAME

env=${6}

# try to activate the env; if this fails, we continue (preferable that we pre-activated)
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "$env" || true
fi

file_path=`sed -n '1p' $2/$3/logs.txt`
protein=`sed -n '2p' $2/$3/logs.txt`    # name of project folder

morgan_directory=`sed -n '4p' $2/$3/logs.txt`

num_molec=`sed -n '8p' $2/$3/logs.txt`

gpu_part=$5

python jobid_writer.py -pt $protein -fp $file_path -n_it $1 -jid $SLURM_JOB_NAME -jn $SLURM_JOB_NAME.txt

echo "Starting Evaluation"
python -u scripts_2/hyperparameter_result_evaluation_pth.py -n_it $1 -d_path $file_path/$protein -mdd $morgan_directory -n_mol $num_molec -ct $4
echo "Creating simple_job_predictions"
python scripts_2/simple_job_predictions_pth.py -pt $protein -fp $file_path -n_it $1 -mdd $morgan_directory -gp $gpu_part -tf_e $env
# the pain of not having slurm: execute them safely in parallel (auto-detect GPU/MIG count)
python -u scripts_2/run_predictions_local.py --jobs-dir "$file_path/$protein/iteration_$1/simple_job_predictions"


# cd $file_path/$protein/iteration_$1/simple_job_predictions/
# echo "running simple_jobs"
# for f in *;do sbatch $f;done