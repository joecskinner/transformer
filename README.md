# The Transformer Architecture

This repository is my personal project to implement the Transformer architecture as described in the 2017 paper ["Attention Is All You Need" by Vaswani et al.](https://arxiv.org/abs/1706.03762), primarily for my own learning and understanding.

## Overview

The aim of this repository is to provide a detailed exploration and implementation of the Transformer model, closely aligned with the design presented in the original paper. It includes:

* [documentation/The Transformer Architecture.md](documentation/The%20Transformer%20Architecture.md): An in-depth guide to the components of the Transformer model and the data pipeline used for training.
* [notebooks/Transformer.ipynb](notebooks/Transformer.ipynb): A Jupyter notebook that demonstrates the training process.


## Use

To run the project on your local machine, follow these steps:

1. Create, activate, and install packages in virtual environment:

    ```
    virtualenv env
    source env/bin/activate 
    pip install -r requirements.txt
    ```

2. Start jupyter lab:

    ```
    jupyter lab
    ```

3. Navigate to [notebooks/Transformer.ipynb](notebooks/Transformer.ipynb) to open and run the notebook.