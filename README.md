# Sentence Embedder Project

## Overview
A PyTorch-based project for representation learning and sentence embedding, featuring a custom Chebyshev extension, Docker integration, and TensorBoard visualization.

## Features
- **Embedder Class:** Enhanced with methods like `embed`, `get_distance`, and more.
- **Chebyshev Extension:** Custom PyTorch extension for advanced computations.
- **Docker Integration:** Containerized environment with TensorBoard support.
- **TensorBoard Visualization:** Real-time monitoring of training metrics.

## Installation

### Prerequisites
- [Docker](https://www.docker.com/get-started)
- NVIDIA drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

### Steps
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your_username/your_repository.git
    cd your_repository
    ```

2. **Build the Docker Image:**
    ```bash
    docker build -t sentence-embedder:latest .
    ```

3. **Run the Docker Container:**
    ```bash
    docker run --gpus all -it --rm \
        -p 6006:6006 \
        -v $(pwd)/logs:/app/logs \
        sentence-embedder:latest
    ```

## Usage

### Running Experiments
```bash
python experiments/main.py --num_epochs 10 --batch_size 128
