# Training a new Pipeline

Training a new coreference resolution pipeline from scratch

First, you need an annotated dataset.

This dataset should contain:

  - The raw text files
  - The annotations minimal infos:
    - The start and end of the annotation (character indexes in the raw text)
    - The label of the annotation (type of entity)
    - The coreference chains ID  

We will make an example by training a pipeline on the [LitBank Dataset](https://github.com/dbamman/litbank)

We first need to download the dataset from the github repository: [https://github.com/dbamman/litbank/tree/master/coref/brat](https://github.com/dbamman/litbank/tree/master/coref/brat)

??? Abstract "**Python Code to Download the Dataset**"
    ```python
            
    import os
    
    local_dataset_path = "datasets/litbank"
    os.makedirs(dataset_path, exist_ok=True)
    
    # The specific GitHub folder you want
    github_folder_path = "coref/brat"
    repo_url = "https://github.com/dbamman/litbank.git"
    
    # Run the git commands (note the exclamation mark !)
    !cd {dataset_path} && git init
    !cd {dataset_path} && git remote add origin -f {repo_url}
    !cd {dataset_path} && git config core.sparseCheckout true
    !cd {dataset_path} && echo "{github_folder_path}/*" >> .git/info/sparse-checkout
    !cd {dataset_path} && git pull origin master
    # Reorganize: move the brat directory to the root of dataset_path, then remove coref and .git
    !cd {dataset_path} && mv coref/brat . && rm -rf coref && rm -rf .git
    
    ```

# Russian

