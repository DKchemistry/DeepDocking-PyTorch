#!/usr/bin/env python3
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('-pt', '--project_name', required=True, help='Name of DD project')
parser.add_argument('-fp', '--file_path', required=True, help='Path to project folder WITHOUT project folder name')
parser.add_argument('-n_it', '--n_iteration', required=True, help='Number of current iteration')
parser.add_argument('-mdd', '--morgan_directory', required=True, help='Path to Morgan fingerprint directory')
parser.add_argument('-gp',  '--gpu_part', required=True, help='name(s) of GPU partitions')
parser.add_argument('-e','--conda_env', required=True, help='name of conda environment')
parser.add_argument('-save','--save_path', required=False, default=None)

io_args = parser.parse_args()
protein   = io_args.project_name
n_it      = int(io_args.n_iteration)
mdd       = io_args.morgan_directory
gpu_part  = str(io_args.gpu_part)
env       = str(io_args.conda_env)

DATA_PATH = os.path.join(io_args.file_path, protein)
SAVE_PATH = io_args.save_path or DATA_PATH

outdir = os.path.join(SAVE_PATH, f'iteration_{n_it}', 'simple_job_predictions')
os.makedirs(outdir, exist_ok=True)

# clean old jobs
for f in glob.glob(os.path.join(outdir, '*')):
    try: os.remove(f)
    except OSError: pass

# one job per shard in the Morgan directory
part_files = sorted(glob.glob(os.path.join(mdd, '*.txt')))
time_str = '00-10:30'  # match prior default

cwd = os.getcwd()
for idx, fpath in enumerate(part_files, start=1):
    job = os.path.join(outdir, f'simple_job_{idx}.sh')
    with open(job, 'w') as ref:
        ref.write('#!/usr/bin/env bash\n')
        ref.write('set -euo pipefail\n')
        ref.write('#SBATCH --ntasks=1\n')
        ref.write('#SBATCH --gres=gpu:1\n')
        ref.write('#SBATCH --cpus-per-task=1\n')
        ref.write('#SBATCH --job-name=phase_5\n')
        ref.write('#SBATCH --mem=0               # memory per node\n')
        ref.write(f'#SBATCH --partition={gpu_part}\n')
        ref.write(f'#SBATCH --time={time_str}\n')
        ref.write(f'export SLURM_JOB_NAME="${{SLURM_JOB_NAME:-phase_5}}"\n')
        ref.write(f'cd "{cwd}/scripts_2"\n')
        ref.write('if command -v conda >/dev/null 2>&1; then\n')
        ref.write('  source "$(conda info --base)/etc/profile.d/conda.sh" || true\n')
        ref.write(f'  conda activate {env} || true\n')
        ref.write('fi\n')
        ref.write('python -u Prediction_morgan_1024.py ')
        ref.write(f'-fn "{os.path.basename(fpath)}" ')
        ref.write(f'-protein "{protein}" ')
        ref.write(f'-it {n_it} ')
        ref.write(f'-mdd "{mdd}" ')
        ref.write(f'-file_path "{SAVE_PATH}"\n')
        ref.write('echo "complete"\n')
    os.chmod(job, 0o755)
