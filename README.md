# cdrpy

Suite of tools for cancer drug response prediction.

## TODO

- [ ] Make tensorflow an optional dependency 

## Setup and Installation

### Linux

#### New GPU Setup

```{shell}
conda create python=3.9.13 --name=cdrpy-tf-gpu-v2

conda activate cdrpy-tf-gpu-v2

module load cuda/11.3 cudnn/8.2.0

conda install -c conda-forge cudatoolkit=11.8.0

pip install nvidia-cudnn-cu11==8.6.0.163

pip install tensorflow==2.11.*

pip install tensorflow-probability==0.19.0
```

#### Old GPU Setup

```{shell}
conda create python=3.9.13 --name=cdrpy-tf-gpu

module load cuda/11.2 cudnn/8.1.1

conda activate cdrpy-tf-gpu

conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0.77

pip install tensorflow==2.10.0
pip install tensorflow-probability==0.18.0
```

#### CPU Setup

Create a conda environments:

```{shell}
conda create python=3.9.13 --name=cdrpy-tf-cpu
```

### MacOS

```{shell}
conda create --name cdrpy-tf python=3.9.13

conda activate cdrpy-tf

python -m pip install tensorflow

python -m pip install tensorflow-metal
```

## Usage

- Currently, scripts must be run from root directory

For GPU usage:

```{shell}
module load cuda/11.2 cudnn/8.1.1

conda activate cdrpy-tf-gpu
```

```{shell}
module load cuda/11.3 cudnn/8.2.0

conda create --name cdrpy-tf-gpu-v2 python=3.9.13
```

## Requirements

- networkx
- pandas
- deepchem
- tensorflow-propability
- scikit-learn
- hydra-core