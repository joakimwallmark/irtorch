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

Ensure you have Python 3.10 or later installed on your system. If not, download and install it from `Python's official website <https://www.python.org/downloads/>`__.

**Installing IRTorch**

You can install IRTorch directly from the Python Package Index (PyPI) or from GitHub.

**Install from PyPI**

Run the following command in your terminal or command prompt: 

.. code-block:: bash

    pip install irtorch

**Install from GitHub**

To install the latest version directly from GitHub, use:

.. code-block:: bash

    pip install git+https://github.com/joakimwallmark/irtorch.git

**Recommended Tools**

When using the package installed on your local machine, consider using an integrated development environment (IDE) such as `Visual Studio Code <https://code.visualstudio.com/>`__ or `PyCharm <https://www.jetbrains.com/pycharm/>`__. Both offer extensive support for Python and include features such as debugging, syntax highlighting, code completion and jupyter notebooks.

- Visual Studio Code: Follow this `tutorial <https://code.visualstudio.com/docs/python/python-tutorial>`__ to set up VSCode for Python development.
- PyCharm: Refer to the `official documentation <https://www.jetbrains.com/help/pycharm/quick-start-guide.html>`__ for installation instructions and getting started.

For jupyter notebooks, refer to `VSCode's <https://code.visualstudio.com/docs/datascience/jupyter-notebooks>`__ or
`PyCharm's <https://www.jetbrains.com/help/pycharm/jupyter-notebook-support.html>`__ documentation.
