import os
import json
from pathlib import Path
import shutil

import velot
from velot.benchmark import benchmark_dotplot
import scanpy as sc
import matplotlib.pyplot as plt

HERE = os.chdir(Path(__file__).resolve().parent)

# Define directories
src_dir = Path("results")
src_dir_old = Path("../../../../../article/benchmark/benchmark_results/real")
real_dir = Path("results/real")
synth_dir = Path("results/synthetic")

# Create destination directories if they don't exist
real_dir.mkdir(parents=True, exist_ok=True)
synth_dir.mkdir(parents=True, exist_ok=True)

# Define dataset categories
real_datasets = ["erythroid", "pancreas", "murine", "hindbrain"]

# Iterate through files in the source directory
for filename in os.listdir(src_dir):
    if not filename.endswith(".json"):
        continue
        
    # Condition 2: Only process files with seed 0
    if "seed0" not in filename:
        continue

    # Extract prefix, dataset, and determine if it's "smooth"
    is_smooth = False
    if "_seed0_raw_" in filename:
        prefix, rest = filename.split("_seed0_raw_")
    elif "_seed0_smooth_" in filename:
        prefix, rest = filename.split("_seed0_smooth_")
        is_smooth = True
    else:
        continue  # Skip if it doesn't match the expected pattern

    # The dataset name is whatever comes after the seed/raw/smooth identifier
    dataset = rest.replace(".json", "")

    # Apply renaming rules for the method
    if prefix == "velot":
        method = "OT"
    elif prefix == "velot_gradient":
        method = "knn"
    else:
        continue # Skip unrecognized prefixes

    # Append _MLP if the original file was "smooth"
    if is_smooth:
        method += "_MLP"

    # Condition 1: Determine the target folder based on dataset name
    if dataset in real_datasets:
        dest_dir = real_dir
    else:
        dest_dir = synth_dir

    # Construct final file paths
    new_filename = f"{method}_{dataset}.json"
    src_path = src_dir / filename
    dest_path = dest_dir / new_filename

    # Read the original JSON
    with open(src_path, "r") as f:
        data = json.load(f)
    
    # Update the 'model' key to the new method name
    if "model" in data:
        data["model"] = method

    # Write the modified JSON to the new destination
    with open(dest_path, "w") as f:
        json.dump(data, f, indent=4)  # indent=4 keeps the JSON human-readable

    print(f"Processed: {filename} -> {dest_dir.name}/{new_filename}")

for file_path in src_dir_old.glob("*.json"):
    # Skip velot results
    if file_path.name.startswith("velot_"):
        continue

    dest_path = real_dir / file_path.name
    
    # Copy the file (use shutil.move instead if you want to remove it from the source folder)
    shutil.copy2(file_path, dest_path)
    print(f"Copied: {file_path.name} -> {real_dir.name}/{file_path.name}")


benchmark_dotplot("results/real", stat="median",
    model_labels={
        "OT": "OT",
        "knn": "knn gradient",
        "OT_MLP": "OT + MLP",
        "knn_MLP": "knn gradient + MLP",
        "scvelo_dynamical": "scVelo (dynamical)",
        "scvelo_stochastic": "scVelo (stochastic)",
        "deepvelo": "DeepVelo",
        "flux_matching": "FluxMatching",
    },
    legend=False, summary=False, include_timing=True,
    show=False, save="figure5_a.png"
    )