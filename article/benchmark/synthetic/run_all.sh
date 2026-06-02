#!/bin/bash

BASE=/home/user/Documents/velot/article/benchmark/synthetic

# PYTHON=/home/lucas/miniforge3/envs/velot/bin/python
# modules=(
#     linear_velot.py
#     bifurcation_velot.py
#     trifurcation_velot.py
#     linear_scvelo_dynamic.py
#     bifurcation_scvelo_dynamic.py
#     trifurcation_scvelo_dynamic.py
#     linear_scvelo_stochastic.py
#     bifurcation_scvelo_stochastic.py
#     trifurcation_scvelo_stochastic.py
# )

# PYTHON=/home/user/miniforge3/envs/deepvelo/bin/python
# modules=(
#     linear_deepvelo.py
#     bifurcation_deepvelo.py
#     trifurcation_deepvelo.py
# )

# PYTHON=/home/user/miniforge3/envs/flux_matching/bin/python
# modules=(
#     linear_flux_matching.py
#     bifurcation_flux_matching.py
#     trifurcation_flux_matching.py
# )

for module in "${modules[@]}"; do
    echo "Running $module"
    $PYTHON "$BASE/$module"
done