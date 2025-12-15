# Training a Mention Detection Model

## Load a dataset

```python

import os, requests, zipfile
from tqdm.auto import tqdm
from collections import Counter

# Where datasets will be stored
local_dataset_directory = "loaded_datasets"
os.makedirs(local_dataset_directory, exist_ok=True)

# Available Datasets: long-litbank-fr-PER-only ; litbank-fr ; litbank

dataset_name = "litbank" #

# Dataset URL
dataset_URL_path = (
    "https://lattice-8094.github.io/propp/datasets/"
    f"{dataset_name}_propp_minimal_implementation.zip"
)

# Local paths
archive_name = dataset_URL_path.split("/")[-1]
archive_path = os.path.join(local_dataset_directory, archive_name)
files_directory = os.path.join(local_dataset_directory, archive_name.replace(".zip", ""))

# Download if needed
if not os.path.exists(archive_path):
    print("Downloading dataset...")
    response = requests.get(dataset_URL_path, stream=True)
    response.raise_for_status()

    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded dataset to {archive_path}")
else:
    print("Archive already exists, skipping download.")

# Extract if needed
if not os.path.exists(files_directory):
    print("Extracting dataset...")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(files_directory)
    print(f"Dataset extracted to {files_directory}")
else:
    print("Dataset already extracted.")

```

## Preprocess the dataset

```python

from pathlib import Path

all_files = sorted(list(set([p.stem for p in Path(files_directory).iterdir() if p.is_file()])))
len(all_files)

```

```python

from propp_fr import load_spacy_model

spacy_model = load_spacy_model("en_core_web_trf")

```

