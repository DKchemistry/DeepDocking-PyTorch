# Background

This is my WIP of a refactor of the original Deep Docking codebase to PyTorch >2.2 based on PhD studies at the Frimurer Group at UCPH and our collaboration with the Gentile Group. The original [Deep Docking](https://github.com/jamesgleave/DD_protocol) GitHub should be used as this code is under active development. If you use Deep Docking in your research, please cite:

Gentile, F. et al. *Deep Docking: A Deep Learning Platform for Augmentation of Structure Based Drug Discovery.* ACS Cent. Sci. 6, 939–949 (2020)  
Gentile, F. et al. *Artificial intelligence–enabled virtual screening of ultra-large chemical libraries with deep docking.* Nat. Protoc. 17, 672–697 (2022)

The documentation for the PyTorch refactor will be updated as soon as possible. Ideally, the platform will retain original behavior wherever possible, with some exceptions. The most dramatic changes are in model training and inference (e.g. phase 4, phase 5): 

* migrations from TensorFlow 1.14/1.15  that used Keras for early stopping or timed stopping, which are now handled in PyTorch. 

* no absolute reliance on SLURM as I do not have easy access to test this, with GPU management by pyNVML. These are custom in nature as they are meant to ease running the software in the context of my group (e.g., they block access to GPUs running Desmond Jobs or ICM jobs, restrict from running on certain GPUs meant to just handle terminal displays)

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

Recently, there was some encouragement to have this implementation functional on our machines across the semi-automated phases and this is currently in progress. 

