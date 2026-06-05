Installation
============

Requirements
------------

- Python ≥ 3.10
- PyTorch ≥ 1.9 (installed separately for CUDA compatibility)

.. note::
   PyTorch is installed separately to allow users to choose the appropriate
   CUDA version for their system. See [pytorch.org](https://pytorch.org)
   for installation options.

Quick install (recommended)
--------------------------

.. code-block:: bash

   # Create environment and install PyTorch first
   conda create -n velot python=3.10 -y
   conda activate velot

   # Install PyTorch (adjust for your CUDA version)
   # GPU:
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # CPU only:
   # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   # Install VelOT
   pip install velot

Developer installation
----------------------

For contributing, modifying the code, or reproducing the paper:

.. code-block:: bash

   # Create environment and install PyTorch first
   conda create -n velot python=3.10 -y
   conda activate velot

   # Install PyTorch (adjust for your CUDA version — see https://pytorch.org)
   # GPU:
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # CPU only:
   # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   # Install VelOT
   git clone https://github.com/lucas-rdlr/velot.git
   cd velot
   pip install -e .

Verify Installation
-------------------

.. code-block:: python

   import velot
   print(velot.__version__)
   # 1.0.0

Dependencies
------------

VelOT automatically installs the following packages:

- ``numpy``, ``scipy``, ``pandas``
- ``anndata``, ``scanpy``
- ``scikit-learn``
- ``matplotlib``
- ``POT`` (Python Optimal Transport)
- ``scvelo``
- ``tqdm``