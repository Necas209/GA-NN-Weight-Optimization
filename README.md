# Genetic Algorithm for Neural Network Weight Optimization

## Description
This GitHub repository provides an implementation of a Genetic Algorithm (GA) for finding optimal weights of a Neural Network (NN). The repository includes our own implementation of the GA algorithm as well as an integration with PyGAD, a popular Python library for Genetic Algorithms. The report can be read at [Neural Network Optimization Using Genetic Algorithms](report.pdf).

## Key Features

- Genetic Algorithm Implementation: Our repository contains a custom implementation of the Genetic Algorithm, specifically designed for optimizing the weights of Neural Networks.
- Neural Network Structure: The repository includes a sample Neural Network structure that can be used as a starting point for optimization. However, users can easily modify the NN architecture according to their specific requirements.
- PyGAD Integration: In addition to our custom implementation, we have also integrated PyGAD into the repository. PyGAD offers a powerful and flexible framework for Genetic Algorithms, allowing users to take advantage of its features and optimizations.
- Training Set Support: The implementation supports the use of a training set for evaluating the fitness of individual solutions during the optimization process.
- Fitness Function Customization: Users can define their own fitness function based on their problem domain and optimization goals.

With this repository, researchers and practitioners can explore the effectiveness of Genetic Algorithms in optimizing Neural Network weights. The code is provided as a starting point for further customization and experimentation, allowing users to fine-tune the GA parameters and explore different configurations.

By leveraging the power of Genetic Algorithms, this repository enables the automated search for optimal weights, potentially improving the performance and accuracy of Neural Networks in various applications.

## Installation

### Python

First, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Then, create a new environment.

```shell
conda create -n torch
```

Activate the environment.

```shell
conda activate torch
```

### PyTorch

Install PyTorch using the [official instructions](https://pytorch.org/get-started/locally/).

### Packages

Install the required packages. These include `jupyter`, `scikit-learn` and `pygad`.
To install these packages, run the following command.

```shell
conda install jupyter scikit-learn
```

```shell
conda install -c conda-forge pygad
```

## Usage

### Jupyter Notebook

To run the Jupyter Notebook, run the following command.

```shell
jupyter notebook
```

And then open the notebook `Genetic Algorithms.ipynb`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
