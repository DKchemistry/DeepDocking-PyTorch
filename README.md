# Background

This is my WIP of a refactor of the original Deep Docking codebase to PyTorch >2.2 based on PhD studies at the [Frimurer Groupand our Computational Chemistry Unit](https://cbmr.ku.dk/research-facilities/ccu/) at the University Of Copenhagen and our collaboration with the [Gentile Group](https://gentilelab.uottawa.ca/) at the University of Ottawa. The original [Deep Docking](https://github.com/jamesgleave/DD_protocol) GitHub should be used as this code is under active development. If you use Deep Docking in your research, please cite:

Gentile, F. et al. *Deep Docking: A Deep Learning Platform for Augmentation of Structure Based Drug Discovery.* ACS Cent. Sci. 6, 939–949 (2020)  
Gentile, F. et al. *Artificial intelligence–enabled virtual screening of ultra-large chemical libraries with deep docking.* Nat. Protoc. 17, 672–697 (2022)

The documentation for the PyTorch refactor will be updated as soon as possible. Ideally, the platform retains original behavior wherever possible, with some exceptions. The most dramatic changes are in model training and inference (e.g. phase 4, phase 5) and workload management: 

* migrations from TensorFlow 1.14/1.15  that used Keras for early stopping or timed stopping, which are now handled in PyTorch. 

* no absolute reliance on SLURM as I do not have easy access to test these implementations, so GPU task management is handled by pyNVML. These are custom in nature as they are meant to ease running the software in the context of my group (e.g., they block access to GPUs running Desmond Jobs or ICM jobs, restrict from running on certain GPUs meant to just handle terminal displays; these may be extended as we test on a variety of machines)

These are primarily tested on NVIDIA A100 80gb GPUs with MIG instances and CUDA Version: 12.8. 

Other changes relate to Schrodinger/Glide related scripts and are minor in scope, reflecting project preferences. Code unrelated to the refactor (e.g FRED specific scripts) or those that we have not generally used will be removed over time to hopefully make following things more straightforward. 

Thank you,

DK

Frimurer Group

## Current Status 

An initial refactor of the training code was done some years ago and compared to a training run we had in the tensorflow version and is available here here: 

```
tf_pth_comparison/2RH1_phase_4_iteration_1.ipynb
```

Professor Gentile and I found the differences in performance minor, though we caution neither runs are seeded. Thereafter we worked on other potential model architectures that are on going. 

Recently, there was some encouragement to have this implementation functional on our machines across the semi-automated phases and this is currently in progress. I hope this maybe useful directly if you are in a similar position with respect to computing environments, or is extensible for your needs! I have tried to keep SLURM headers in the shell scripts, but their useability is not tested. Generally, they are bypassed, and `jobid_writer.py` is given a "dummy" argument such that it runs. 

## Documentation 

Hopefully, I can upload a pre-prepared library that we have enumerated that was used in our earlier implentation of tensorflow code, that is currently being used in testing the pytorch implementation. The project directory that my current testing corresponds too. Briefly, we used an initial ~35M diversity subset of Enamine REAL Lead-Like, which totals to 66M post-library preparation steps as described on the original Deep Docking GitHub. Our  library preparation and fingerprinting are unchanged from the original implementation with respect to intent. Though, due licensing differences, alternative software is used in [stereochemical enumeration](http://www.mayachemtools.org/docs/scripts/html/RDKitEnumerateStereoisomers.html) and [protonation](https://www.schrodinger.com/platform/products/epik/). 

The following documentation contains examples that reference this library and gives the arguments I have used as examples to hopefully make using the code more straightforward. 

`phase_1.sh` is unchanged with respect to the underlying scripts, just bypasses SLURM.

```sh
conda activate pth_dd
export SLURM_JOB_NAME=phase_1 # passed to the jobid_writer.py
bash ./phase_1.sh <iteration_number> <number_of_threads> <absolute_path_to_project_directory> <name_of_project> <mols_to_sample> <conda env> 
```

Example: 

```sh
conda activate pth_dd
export SLURM_JOB_NAME=phase_1
bash ./phase_1.sh 1 120 /mnt/data/dk/work/DeepDocking/projects Manuscript_pytorch_2RH1 46200 pth_dd
```

Remember that the logs.txt file should be appropriately populated in your <name_of_project> directory!

`phase_2_glide.sh` is mostly unchanged, though the LigPrep command in this file is attempting to produce only the most dominant protomer/tautomer via input SMILES. Your usage should likely reflect your preferences. Additionally, the thread count is the *total* given to LigPrep, given three files (test/train/valid) it will divide by 3 for the splits and execute in parallel, Schrodinger job manager handles CPU task management. The command also executes when run, so be kind to your colleagues and check how many threads are in use.

```sh
conda activate pth_dd
SLURM_JOB_NAME=phase_2  # passed to the jobid_writer.py
bash ./phase_2_glide.sh  <iteration_number> <total_threads_for_ligprep> <absolute_path_to_project_directory> <name_of_project>
```

Example:

```sh
conda activate pth_dd
SLURM_JOB_NAME=phase_2 
bash ./phase_2_glide.sh 1 120 /mnt/data/dk/work/DeepDocking/projects Manuscript_pytorch_2RH1
```

Here, the LigPrep commands that are generated give each individual set 40 threads. 

`phase_3_glide.sh` utilizies the template found from the logs.txt file of your project (line 9). It will also divide the <total_threads_for_glide> for the amount of threads to supply for each job and it will triple the count of `-NJOBS`. This is not always the most optimal way to run the command in my experience, so these arguments might change. The command also executes when run, so be kind to your colleagues and check how many threads are in use.

```sh
conda activate pth_dd
SLURM_JOB_NAME=phase_3
bash ./phase_3_glide.sh <iteration> <total_threads_for_glide> <absolute_path_to_project_directory> <name_of_project>
```

Example: 

```sh
conda activate pth_dd
SLURM_JOB_NAME=phase_3
bash ./phase_3_glide.sh 1 150 /mnt/data/dk/work/DeepDocking/projects Manuscript_pytorch_2RH1
```

Here, each test/train/valid docking job will get `localhost:50` and `-NJOBS 150`. I have encountered one split finishing much faster than others despite equal splitting in the number of ligands to dock, I hadn't experienced this before, so it may be more prudent to set this up differently. 

`phase_4_pth.sh` is fairly different as it now relies on pytorch model and associated code. The output is mostly the same. Tensorboard logging is also enabled, which is a hold-over from experimenting with different model architectures. Some efforts have been made to simply some of the associated or unused code for readability/maintainability, though this is in progress. The output of training mirrors the equivalent of the prior tensorflow versions and is used for inference in phase 5. The script does not automatically run the `simple_jobs_N.sh` scripts created currently, relying on the user to do so from that from `interation_n/simple_job` directory. This will likely change later. The arugments related to slurm are "dummy" arguments to keep other processes happy, as SLURM is not being used here. 

```sh
conda activate pth_dd
SLURM_JOB_NAME=phase_4
bash ./phase_4_pth.sh <iteration_number> <threads> <absolute_path_to_project_directory> <name_of_project> <gpu_partition> <total_iterations> <percent_first_mols> <percent_last_mols> <recall_value> <time_string> <conda_env>
```

Example:

```sh
conda activate pth_dd
SLURM_JOB_NAME=phase_4
bash ./phase_4_pth.sh 1 3 /mnt/data/dk/work/DeepDocking/projects Manuscript_pytorch_2RH1 dummy 10 1 0.01 0.9 00-15:00 pth_dd
```

(typically its 11 iterations)

`phase_5_pth.sh` is also quite different and is functional in terms of output, though the manner in which GPU inference is distrbuted is fairly conservative and assumes the conventional 1M member library splitting as implemented originallly in Deep Docking. I find it useful in testing currently but it would likely need to be amended in the future, as my testing has been in ~70M range for the fingerprinted library and prospective is more helpful on the >1B scale. Currently, this does execute GPU training on launch. 

```sh
conda activate pth_dd
SLURM_JOB_NAME=phase_5
bash ./phase_5_pth.sh <iteration_number> <absolute_path_to_project_directory> <name_of_project> <recall_value> <gpu_partition> <conda_env>
```

Example: 

```sh
conda activate pth_dd
SLURM_JOB_NAME=phase_5
bash ./phase_5_pth.sh 1 /mnt/data/dk/work/DeepDocking/projects Manuscript_pytorch_2RH1 0.9 ignore pth_dd
```

Testing for further iterations continues, incrementing the iteration number behaves (1 -> 2, etc) so far as expected. While comparison to previous tensorflow versions is difficult, general behavior seems comparable, with median score improvements in line with previous experience. 

The final extraction phase has not yet been refactored. 