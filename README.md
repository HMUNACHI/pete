# Tiny Attention Networks (TANs)

This repository implements Tiny Attention Networks (TANs)

## GPU server checklist (Codes are for Ubuntu Linux)
- Should have a working nvcc compiler (verify with `nvcc --version`, install with `apt install nvidia-cuda-toolkit`)
- We used `Cuda-tool-kit 11.7` which is compatible with the pytorch version in `environment.yaml`
- Should have g++ compiler (verify with `which g++`, install with `sudo apt install build-essential`)
- Chebychev expansion itself is written in C++/CUDA, a GPU is compulsory

## How to use
- `git clone https://github.com/HMUNACHI/tiny-attention-networks.git && cd tiny-attention-networks`
- Create the environment with the necessary packages `conda env create -f environment.yml`
- Activate with `conda activate tan_env`
- Install the custom gpu kernels with `cd src/polynomial_embeddings && python setup.py install && cd ../..`
- Run simple tiny example with `python main.py`, check main.py for arguments.
- Tensorbnoard will launch after training.