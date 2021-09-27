# Introduction

VI-HDS-TORCH is the PyTorch version of VI-HDS, which was released in conjunction with an [ICML 2019](https://icml.cc) paper, [Efficient Amortised Bayesian Inference for Hierarchical and Nonlinear Dynamical Systems](https://arxiv.org/abs/1905.12090). 

VI-HDS is a a flexible, scalable Bayesian inference framework for nonlinear dynamical systems characterised by distinct and hierarchical variability at the individual, group, and population levels. 
We cast parameter inference as stochastic optimisation of an end-to-end differentiable, block-conditional variational autoencoder. 
We specify the dynamics of the data-generating process as an ordinary differential equation (ODE) such that both the ODE and its solver are fully differentiable. 
This model class is highly flexible: the ODE right-hand sides can be a mixture of user-prescribed or “white-box” sub-components and neural network or “black-box” sub-components. 
Using stochastic optimisation, our amortised inference algorithm could seamlessly scale up to massive data collection pipelines (common in labs with robotic automation).

## Citation
If you use this code or build upon it, please use the following (bibtex) citation:
```bibtex
@InProceedings{
	title = "Efficient Amortised Bayesian Inference for Hierarchical and Nonlinear Dynamical Systems",
	author = "Geoffrey Roeder and Paul K Grant and Andrew Phillips and Neil Dalchau and Edwards Meeds",
	booktitle = "International Conference on Machine Learning (ICML 2019)",
	year = "2019"
}
```

# Dependencies
- [PyTorch](https://www.pytorch.org/), a deep learning framework
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq), differentiable ODE solvers with full GPU support and O(1)-memory backpropagation.
- [NumPy](http://www.numpy.org/), numerical linear algebra for Python
- [Pandas](https://pandas.pydata.org/), data analysis and data structures
- [Seaborn](https://seaborn.pydata.org/index.html), a statistical data visualization library. 
- [Tensorboard](https://www.tensorflow.org/tensorboard/), a visualization tool for tensors.
- [CUDA](https://developer.nvidia.com/cuda-zone), a parallel computing  framework. It's not essential, as the code can run (albeit, more slowly) in CPU mode.

We have supplied a requirements file for installing the project dependencies with pip. 
    
    pip install -r requirements.txt

For Anaconda, you can use the `environment.yml` file by simply calling:
    
    conda env create

which creates an environment called `vi-hds`.


# Running an example

1. Ensure the `.` directory is on your python path. Also, optionally, set the environment variables INFERENCE_DATA_DIR and INFERENCE_RESULTS_DIR to the directories to which data will be read and results will be written, and export it. By default, these are set to local paths "data" (built-in data files are stored here) and "results" (already in the .gitignore file) respectively.

    In Linux:
    ```bash
    export PYTHONPATH=.
    export INFERENCE_DATA_DIR=data
    export INFERENCE_RESULTS_DIR=results
    ```

    In Windows:
    ```dos
    set PYTHONPATH=.
    set INFERENCE_DATA_DIR=data
    set INFERENCE_RESULTS_DIR=results
    ```

2. Run the `dr_constant_icml` example by calling: 

    ```bash
    python vihds/run_xval.py --experiment=EXAMPLE specs/dr_constant_icml.yaml 
    ```

3. Run tensorboard to visualise the output. A folder will be created in your user-specified results directory with a name that combines the EXAMPLE name and a timestamp. E.g.

    `tensorboard --logdir=EXAMPLE_20181123T174132369485`

    TensorBoard uses port 6006 by default, so you can then visualise your example at http://localhost:6006. Alternatively, you can specify another port.

# Running tests

We make use of the [pytest](https://pytest.org/) library to run tests. 
    
In Windows:
```dos
set PYTHONPATH=.
pytest tests
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
