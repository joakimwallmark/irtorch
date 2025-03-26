FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Set up a non-interactive environment for apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Update and install Python, pip, and git as root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        openssh-client \
        make \
        pandoc && \
    apt-get clean && rm -rf /var/lib/apt/lists/* \

# Create a non-root user "vscode" and set up its home directory
RUN useradd -ms /bin/bash vscode && \
    mkdir -p /home/vscode && \
    chown -R vscode:vscode /home/vscode

# Configure Git to handle line endings
RUN git config --global core.autocrlf input

# Switch to the "vscode" user for subsequent commands
USER vscode
WORKDIR /home/vscode

# Install PyTorch with CUDA 12.6 support, pytest, JupyterLab, and ipykernel
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir torch pytest==7.2.0 pytest-cov jupyterlab ipykernel sphinx sphinx_rtd_theme nbsphinx sphinxcontrib.bibtex --extra-index-url https://download.pytorch.org/whl/cu126
