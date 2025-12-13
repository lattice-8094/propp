# Training a new Pipeline

Training a new coreference resolution pipeline from scratch

## Dataset Preparation

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
We can now take a look at the dataset:

??? Abstract "Python Code"
    ```python

    from pathlib import Path
    from collections import Counter
    
    raw_files_path = "datasets/litbank/tsv"
    
    all_files = [p for p in Path(raw_files_path).iterdir() if p.is_file()]
    
    extensions = dict(Counter(p.suffix for p in files).most_common())
    print(f"The dataset contains {len(all_files):,} files with the following extensions:")
    for extension, count in extensions.items():
        print(f"\t'{extension}' = {count:,}")
    ```

```
The dataset contains 200 files with the following extensions:
	'.txt' = 100
	'.ann' = 100
```

While the `.txt` file contains the raw text, we need to take a closer look at the `.ann` file.

??? Abstract "Python Code"
    ```python

    txt_files = [p for p in all_files if p.suffix == ".txt"]
    with open(ann_files[0], "r", encoding="utf-8") as f:
            text = f.read()
            
    ann_files = [p for p in all_files if p.suffix == ".ann"]
    ann_df = pd.read_csv(ann_files[0], sep="\t", header=None)
    
    print(ann_df.sample(5).to_markdown())
    ```

|     | 0       | 1    | 2             |   3 |   4 |   5 | 6        | 7   | 8    |
|----:|:--------|:-----|:--------------|----:|----:|----:|:---------|:----|:-----|
| 393 | COREF   | T174 | Ralph_Percy-0 | nan | nan | nan | nan      | nan | nan  |
|  18 | MENTION | T27  | 15            |  45 |  15 |  46 | my house | FAC | NOM  |
|  42 | MENTION | T65  | 70            |   4 |  70 |   4 | wife     | PER | NOM  |
| 335 | COREF   | T135 | settlers-7    | nan | nan | nan | nan      | nan | nan  |
| 245 | MENTION | T254 | 51            |   0 |  51 |   0 | I        | PER | PRON |


We see that the first column contains the annotation type, on which the content of other columns depends.

??? Abstract "Python Code"
    ```python
    
    type_to_description_mapping = {
        "MENTION": "Entity mention with span and semantic type",
        "COREF": "True coreference link between two mentions",
        "COP": "Copular relation between two mentions (not treated as coreference)",
        "APPOS": "Appositional relation between two mentions (not treated as coreference)",
    }
    
    row_types_df = pd.DataFrame(
            Counter(ann_df[0]).most_common(),
            columns=["Column '0' Type", "Count"]
        )
    row_types_df["Row Type"] = row_types_df["Column '0' Type"].map(type_to_description_mapping)
    
    print(row_types_df.to_markdown(index=False))
    ```

| Column '0' Type   |   Count | Row Type                                                                |
|:------------------|--------:|:------------------------------------------------------------------------|
| MENTION           |     316 | Entity mention                                                          |
| COREF             |     307 | True coreference link between two mentions                              |
| COP               |       6 | Copular relation between two mentions (not treated as coreference)      |
| APPOS             |       3 | Appositional relation between two mentions (not treated as coreference) |

In this case, only the `MENTION` and `COREF` annotations are relevant. 

We will create two new objects for both the mentions and the coreference links:



















## Russian

🚧 Coming soon... 🚧