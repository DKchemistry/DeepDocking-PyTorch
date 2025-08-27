# Status

This is my WIP of a refactor of the original Deep Docking codebase to PyTorch >2.2 based on PhD studies at the Frimurer Group at UCPH and our collaboration with the Gentile Group. The original [Deep Docking](https://github.com/jamesgleave/DD_protocol) GitHub should be used as this code is under active development. If you use Deep Docking in your research, please cite:

Gentile, F. et al. *Deep Docking: A Deep Learning Platform for Augmentation of Structure Based Drug Discovery.* ACS Cent. Sci. 6, 939–949 (2020)  
Gentile, F. et al. *Artificial intelligence–enabled virtual screening of ultra-large chemical libraries with deep docking.* Nat. Protoc. 17, 672–697 (2022)

The documentation for the PyTorch refactor will be updated as soon as possible. Ideally, the platform will retain original behavior wherever possible, with some exceptions. It should support HPC Linux Enironments and local development (e.g., alternative architectures) via Apple M series GPUs and not require SLURM as a workload manager, due to to the computing environments we use at our group. It is functionally specific to our computing environments and scoring enginges, but is extensible by community efforts. 

Thank you,
DK
Frimurer Group


