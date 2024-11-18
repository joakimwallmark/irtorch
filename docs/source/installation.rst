Installation
==================

This guide provides instructions for installing the IRTorch package either on your local machine or running it from a Jupyter notebook in a cloud environment.

Install on your own machine
---------------------------
**Prerequisites**

Ensure you have Python 3.10 or later installed on your system. If not, download and install it from `Python's official website <https://www.python.org/downloads/>`__. We suggest you check the option to add python.exe to PATH during installation to be able to run Python from the command line.

**Installing IRTorch**

If you have a compatible NVIDIA GPU and want to leverage GPU acceleration for better performance, you need to install the GPU-enabled version of PyTorch before installing IRTorch. Instructions can be found on the `PyTorch website <https://pytorch.org/get-started/locally/>`__. After installing PyTorch, or if you don't have a compatible GPU, you can 
install IRTorch directly from the Python Package Index (PyPI) or from GitHub as shown below.

**Install IRTorch from PyPI**

Run the following command in your terminal or command prompt: 

.. code-block:: bash

    pip install irtorch

**Install IRTorch from GitHub**

To install the latest version directly from GitHub, use:

.. code-block:: bash

    pip install git+https://github.com/joakimwallmark/irtorch.git

**Recommended Tools**

You can run Python code and **IRTorch** directly from the terminal with python installed. There are also several ways to run **IRTorch** within Jupyter Notebooks on your own computer. One of the simplest ways is to use `JupyterLab <https://jupyter.org/>`__, which is a web-based interactive development environment where you can work with Jupyter notebooks, code, and data.

1. Install the **jupyterlab** package:

.. code-block:: bash

    pip install jupyterlab

2. Start a JupyterLab session:

.. code-block:: bash

    jupyter lab

3. Create a new notebook and import the **IRTorch** package to start using it:

.. code-block:: python

    import irtorch
    print("IRTorch successfully installed")

Run from a Cloud Environment
----------------------------
To use the package in a cloud-based Jupyter notebook environment such as Google Colab, follow these steps:

1. Open a new Jupyter notebook on `Google Colab <https://colab.research.google.com/>`__.
2. Change to a GPU enabled runtime by selecting `Runtime` -> `Change runtime type` -> `Hardware accelerator` -> `GPU`. (Recommended but not required)
3. Install the package by running the following command in a new cell:

.. code-block::

    !pip install irtorch

4. Verify the installation by importing the package in another cell:

.. code-block:: python

    import irtorch
    print("IRTorch successfully installed")

See the examples section and the package API reference for more information on how to use the package.
