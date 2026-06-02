VelOT Documentation
====================

**VelOT** (Velocity via Optimal Transport) is a kinetic-free framework
for estimating RNA velocity from single-cell transcriptomic data.

Instead of modeling the molecular kinetics of splicing and degradation,
VelOT treats velocity inference as an optimal transport problem in gene
expression space, producing smooth, continuous velocity fields without
requiring spliced/unspliced count matrices.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/erythroid_tutorial
   tutorials/pancreas_tutorial

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   methodology

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/pp
   api/tl
   api/pl
   api/metrics
   api/datasets
   api/benchmark

.. toctree::
   :maxdepth: 1
   :caption: About

   changelog
   citation

If you use **VelOT** in your research, please cite:

.. code-block:: bibtex

   @article{velot2026,
   title   = {VelOT: Velocity via Optimal Transport for kinetic-free
               RNA velocity estimation},
   author  = {Rincon de la Rosa, Lucas},
   journal = {bioRxiv},
   year    = {2026},
   doi     = {10.1101/2025.XX.XX.XXXXXX}
   }