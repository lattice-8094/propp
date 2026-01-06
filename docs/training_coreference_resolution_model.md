# Training a Coreference Resolution Model

## Load a dataset

Select a dataset annotated with Coreference. See [Available Dataset](available_datasets/#coreference-resolution-propp-formatted-datasets).

```python

import os, requests, zipfile
from tqdm.auto import tqdm
from collections import Counter

# Where datasets will be stored
local_dataset_directory = "loaded_datasets"
os.makedirs(local_dataset_directory, exist_ok=True)

# Available Datasets: long-litbank-fr-PER-only ; litbank-fr ; litbank ; conll2003-NER

dataset_name = "conll2003-NER" #

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

## Load the spaCy model

Here, as the dataset is in English, and we have a GPU available, we will use the `en_core_web_trf` model.

```python

from propp_fr import load_spacy_model

spacy_model = load_spacy_model("en_core_web_trf")

```

Define Preprocessing Functions

```python

def realign_tokens_offsets(tokens_df, entities_df):
    start_tokens = []
    end_tokens = []
    new_byte_onsets = []
    new_byte_offsets = []

    for mention_byte_onset, mention_byte_offset in entities_df[["byte_onset", "byte_offset"]].values:
        start_token = tokens_df[tokens_df["byte_offset"] > mention_byte_onset].index.min()
        end_token = tokens_df[tokens_df["byte_onset"] < mention_byte_offset].index.max()
        new_byte_onsets.append(tokens_df.loc[start_token, "byte_onset"])
        new_byte_offsets.append(tokens_df.loc[end_token, "byte_offset"])

        start_tokens.append(start_token)
        end_tokens.append(end_token)

    entities_df["start_token"] = start_tokens
    entities_df["end_token"] = end_tokens
    entities_df["byte_onset"] = new_byte_onsets
    entities_df["byte_offset"] = new_byte_offsets

    return entities_df

def extract_mention_text(text_content, entities_df):
    mention_texts = []
    for mention_byte_onset, mention_byte_offset in entities_df[["byte_onset", "byte_offset"]].values:
        mention_texts.append(text_content[mention_byte_onset:mention_byte_offset])
    entities_df["text"] = mention_texts
    entities_df["text"] = entities_df["text"].astype(str)
    return entities_df

```

Preprocess the dataset to make it ready for mention spans detection training.

```python

from propp_fr import load_text_file, generate_tokens_df, save_tokens_df
from propp_fr import load_tokens_df, load_entities_df, add_features_to_entities, save_entities_df

for file_name in tqdm(all_files):
    if os.path.exists(os.path.join(files_directory, file_name + ".tokens")):
        continue
    text_content = load_text_file(file_name, files_directory)
    tokens_df = generate_tokens_df(text_content, spacy_model, verbose=0)
    entities_df = load_entities_df(file_name, files_directory)

    entities_df = realign_tokens_offsets(tokens_df, entities_df)
    entities_df = extract_mention_text(text_content, entities_df)

    entities_df = add_features_to_entities(entities_df, tokens_df)
    entities_df["gender"] = "Not_Assigned" # Not available for english
    entities_df["number"] = "Not_Assigned" # Not available for english
    entities_df["grammatical_person"] = "4" # Not available for english

    save_entities_df(entities_df, file_name, files_directory)
    save_tokens_df(tokens_df, file_name, files_directory)

```

```python

from propp_fr import mentions_detection_LOOCV_full_model_training, generate_NER_model_card_from_LOOCV_directory

from collections import Counter

NER_categories = []
for file_name in all_files:
    entities_df = load_entities_df(file_name, files_directory)
    NER_categories.extend(entities_df["cat"].tolist())
print(Counter(NER_categories))
NER_cat_list = list(set(NER_categories))
print(NER_cat_list)

```

## Define Training Parameters

```python

subword_pooling_strategy = "first_last"
nested_levels = [0]
tagging_scheme = "BIOES"
test_split = [file for file in all_files if file.startswith("test")]
print(f"Test split: {len(test_split):,}")

model_name = "FacebookAI/roberta-large"
embedding_model_name = model_name.split("/")[-1]
trained_model_directory = os.path.join(files_directory, f"mentions_detection_model_{embedding_model_name}")

```

## Train and Evaluate the model And Generate the Model Card

```python

mentions_detection_LOOCV_full_model_training(files_directory=files_directory,
                                             trained_model_directory=trained_model_directory,
                                             model_name=model_name,
                                             subword_pooling_strategy=subword_pooling_strategy,
                                             nested_levels=nested_levels,
                                             NER_cat_list=NER_cat_list,
                                             tagging_scheme=tagging_scheme,
                                             train_final_model=False,
                                             files_to_use_in_cross_validation=[test_split],
                                             verbose=0)

generate_NER_model_card_from_LOOCV_directory(trained_model_directory)

```



??? Abstract "Model Card Example"

    ---
    language: fr
    tags:
    - NER
      - literary-texts
      - nested-entities
      - propp-fr
      license: apache-2.0
      metrics:
      - f1
      - precision
      - recall
      base_model:
      - FacebookAI/roberta-large
      pipeline_tag: token-classification
    ---
    
    ## INTRODUCTION:
    This model, developed as part of the [propp-fr project](https://lattice-8094.github.io/propp/), is a **NER model** built on top of [roberta-large](https://huggingface.co/FacebookAI/roberta-large) embeddings, trained to predict nested entities in french, specifically for literary texts.
    
    The predicted entities are:
    
      - PER: Person names (real or fictional)
        - ORG: Organizations (companies, institutions)
        - LOC: Geographical locations (non-political: mountains, rivers, cities)
        - MISC: Miscellaneous entities (events, nationalities, products, etc.)
    
    ## MODEL PERFORMANCES (TEST SET):
    | NER_tag   | precision   | recall   | f1_score   | support   | support %   |
    |-----------|-------------|----------|------------|-----------|-------------|
    | LOC       | 93.74%      | 93.41%   | 93.57%     | 1,668     | 29.53%      |
    | ORG       | 92.98%      | 93.26%   | 93.12%     | 1,661     | 29.41%      |
    | PER       | 97.90%      | 98.02%   | 97.96%     | 1,617     | 28.63%      |
    | MISC      | 81.33%      | 81.91%   | 81.62%     | 702       | 12.43%      |
    | micro_avg | 93.16%      | 93.25%   | 93.21%     | 5,648     | 100.00%     |
    | macro_avg | 91.49%      | 91.65%   | 91.57%     | 5,648     | 100.00%     |
    
    ## TRAINING PARAMETERS:
    - Entities types: ['MISC', 'ORG', 'PER', 'LOC']
      - Tagging scheme: BIOES
      - Nested entities levels: [0]
      - Split strategy: Leave-one-out cross-validation (1393 files)
      - Train/Validation split: 0.85 / 0.15
      - Batch size: 16
      - Initial learning rate: 0.00014
    
    ## MODEL ARCHITECTURE:
    Model Input: Maximum context roberta-large embeddings (1024 dimensions)
    
    - Locked Dropout: 0.5
    
      - Projection layer:
        - layer type: highway layer
        - input: 1024 dimensions
        - output: 2048 dimensions
    
      - BiLSTM layer:
        - input: 2048 dimensions
        - output: 256 dimensions (hidden state)
    
      - Linear layer:
        - input: 256 dimensions
        - output: 17 dimensions (predicted labels with BIOES tagging scheme)
    
      - CRF layer
    
    Model Output: BIOES labels sequence
    
    ## HOW TO USE:
    [Propp Documentation](https://lattice-8094.github.io/propp/quick_start/)
    
    ## PREDICTIONS CONFUSION MATRIX:
    | Gold Labels   | LOC   | ORG   | PER   |   MISC |   O | support   |
    |---------------|-------|-------|-------|--------|-----|-----------|
    | LOC           | 1,558 | 38    | 1     |     22 |  49 | 1,668     |
    | ORG           | 30    | 1,549 | 2     |     12 |  68 | 1,661     |
    | PER           | 6     | 7     | 1,585 |      2 |  17 | 1,617     |
    | MISC          | 25    | 24    | 10    |    575 |  68 | 702       |
    | O             | 43    | 48    | 21    |     96 |   0 | 208       |
