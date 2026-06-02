#!/bin/bash

BASE=/home/user/Documents/velot/article/benchmark/real

# PYTHON=/home/lucas/miniforge3/envs/deepvelo/bin/python
# modules=(
#     erythroid_velot.py
#     murine_velot.py
#     hindbrain_velot.py
#     pancreas_velot.py
#     erythroid_scvelo_dynamic.py
#     murine_scvelo_dynamic.py
#     hindbrain_scvelo_dynamic.py
#     pancreas_scvelo_dynamic.py
#     erythroid_scvelo_stochastic.py
#     murine_scvelo_stochastic.py
#     hindbrain_scvelo_stochastic.py
#     pancreas_scvelo_stochastic.py
# )

# PYTHON=/home/user/miniforge3/envs/deepvelo/bin/python
# modules=(
#     erythroid_deepvelo.py
#     murine_deepvelo.py
#     hindbrain_deepvelo.py
#     pancreas_deepvelo.py
# )

# PYTHON=/home/user/miniforge3/envs/flux_matching/bin/python
# modules=(
#     erythroid_flux_matching.py
#     murine_flux_matching.py
#     hindbrain_flux_matching.py
#     pancreas_flux_matching.py
# )

for module in "${modules[@]}"; do
    echo "Running $module"
    $PYTHON "$BASE/$module"
done