Installation
============

Requirements
------------

- Python ≥ 3.9
- PyTorch ≥ 1.9 (installed separately for CUDA compatibility)

Install from GitHub
-------------------

We recommend a fresh conda environment:

.. code-block:: bash

   # Create environment
   conda create -n velot python=3.10 -y
   conda activate velot

   # Install PyTorch first (adjust for your CUDA version)
   # GPU:
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # CPU only:
   # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   # Install VelOT
   git clone https://github.com/lucas-rdlr/velot.git
   cd velot
   pip install -e .

.. note::

   PyTorch is installed separately to allow users to choose the
   appropriate CUDA version for their system. See
   `pytorch.org <https://pytorch.org>`_ for installation options.

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