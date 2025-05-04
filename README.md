# Parameter-Efficient Transformer Embeddings (PETE)

## GPU server checklist (Codes are for Ubuntu Linux)
- Should have a working nvcc compiler (verify with `nvcc --version`, install with `apt install nvidia-cuda-toolkit`)
- Should have g++ compiler (verify with `which g++`, install with `sudo apt install build-essential`)
- Chebychev expansion itself is written in C++/CUDA, a GPU is compulsory

## How to use
- `git clone https://github.com/HMUNACHI/tiny-attention-networks.git && cd tiny-attention-networks`
- Create the environment with the necessary packages `conda env create -f environment.yml`
- Activate with `conda activate pete_env`
- Install cuda toolkit in the environment `conda install -c nvidia/label/cuda-11.7.0 cuda-toolkit=11.7 cuda-nvcc=11.7`
- Install the custom kernels with `cd polynomial_embeddings && pip install . && cd ..`
- Run simple tiny example with `python train.py`, check train.py for arguments.
- Tensorbnoard will launch after training.

The base implementation is a transformer with Fourier embeddings, for the much smaller model (called PETE), change the `is_pete=True` here in the MLP class or call with it.