# cdrpy

Suite of tools for cancer drug response prediction.

## Setup and Installation

### GPU Setup

```{shell}
conda create python=3.9.13 --name=cdrpy-tf-gpu

module load cuda/11.2 cudnn/8.1.1

conda activate cdrpy-tf-gpu

conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0.77

pip install tensorflow==2.10.0
pip install tensorflow-probability==0.18.0
```

### CPU Setup

Create a conda environments:

```{shell}
conda create python=3.9.13 --name=cdrpy-tf-cpu
```

## Usage

- Currently, scripts must be run from root directory
  
For GPU usage:

```{shell}
module load cuda/11.2 cudnn/8.1.1

conda activate cdrpy-tf-gpu
```