Installation
==================

This guide provides instructions for installing the IRTorch package either on your local machine or running it from a Jupyter notebook in a cloud environment.

Run from a Cloud Environment
----------------------------
To use the package in a cloud-based Jupyter notebook environment such as Google Colab, follow these steps:

1. Open a new Jupyter notebook on `Google Colab <https://colab.research.google.com/>`__.
2. Install the package by running the following command in a new cell:

.. code-block::

    !pip install irtorch

3. Verify the installation by importing the package in another cell:

.. code-block:: python

    import irtorch
    print("IRTorch successfully installed")

Install on your own machine
---------------------------
**Prerequisites**

Ensure you have Python 3.10 or later installed on your system. If not, download and install it from `Python's official website <https://www.python.org/downloads/>`__. We suggest you check the option to add python.exe to PATH during installation to be able to run Python from the command line.

**Installing IRTorch**

You can install **IRTorch** directly from the Python Package Index (PyPI) or from GitHub.

**Install from PyPI**

Run the following command in your terminal or command prompt: 

.. code-block:: bash

    pip install irtorch

**Install from GitHub**

To install the latest version directly from GitHub, use:

.. code-block:: bash

    pip install git+https://github.com/joakimwallmark/irtorch.git

**Recommended Tools**

You can run Python code and **IRTorch** directly from the terminal with python installed. There are also several ways to run **IRTorch** within Jupyter Notebooks on your own computer using the `Jupyter project <https://jupyter.org/>`__. To install the **jupyterlab** package, run:

.. code-block:: bash

    pip install jupyterlab

Then start a Jupyter notebook server by running:

.. code-block:: bash

    jupyter lab

Now simply select a notebook kernel and start using the package.

Of course, you can also run notebooks or python code through `Anaconda <https://docs.jupyter.org/en/latest/install/notebook-classic.html>`__, `Visual Studio Code <https://code.visualstudio.com/docs/datascience/jupyter-notebooks>`__ or `PyCharm's <https://www.jetbrains.com/help/pycharm/jupyter-notebook-support.html>`__.