Contributing
============

We welcome contributions to VelOT! This page describes how to set up a
development environment and submit changes.

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/lucas-rdlr/velot.git
   cd velot
   conda create -n velot-dev python=3.10 -y
   conda activate velot-dev

   # Install PyTorch (adjust for your CUDA version)
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # Install VelOT in editable mode with dev dependencies
   pip install -e ".[dev,docs]"

Running Tests
-------------

.. code-block:: bash

   pytest tests/ -v

Code Style
----------

We follow standard Python conventions:

- Type hints for all public functions
- NumPy-style docstrings
- Line length ≤ 88 characters (Black formatter)

.. code-block:: bash

   black velot/
   flake8 velot/

Building Documentation Locally
-------------------------------

.. code-block:: bash

   cd docs
   make html
   # Open _build/html/index.html in your browser

Reporting Issues
----------------

Please report bugs and feature requests on the
`GitHub issue tracker <https://github.com/lucas-rdlr/velot/issues>`_.

Submitting Changes
------------------

1. Fork the repository
2. Create a feature branch (``git checkout -b feature/my-feature``)
3. Commit your changes with clear messages
4. Push to your fork and open a Pull Request
5. Describe what your PR does and link any relevant issues

.. note::

   For major changes, please open an issue first to discuss the
   proposed modification.