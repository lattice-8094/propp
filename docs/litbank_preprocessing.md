# LitBank Preprocessing

Tutorial on how to preprocess the [LitBank Dataset](https://github.com/dbamman/litbank) for coreference resolution.
    
This dataset should contain:

  - The raw text files
  - The annotations minimal infos:
    - The start and end of the annotation (character indexes in the raw text)
    - The label of the annotation (type of entity)
    - The coreference chains ID  

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
    
    all_files = sorted([p for p in Path(raw_files_path).iterdir() if p.is_file()])
    
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
    
    file_name = all_files[0].stem

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

In this case, only the `MENTION` and `COREF` annotations are relevant, as copulae and appositions are not considered strict coreference relations.

We merge `MENTION` and `COREF` annotations into a single table and assign unique coreference IDs to singleton mentions:

??? Abstract "Python Code"
    ```python
    
    # Filter mentions
    mention_df = ann_df[ann_df[0]=="MENTION"].reset_index(drop=True)
    mention_df = mention_df[[1,2,3,5,6,7,8]]
    mention_df.columns = ["mention_ID","paragraph_ID","start_token_within_sentence","end_token_within_sentence","text","cat","prop"]
    mention_df[["paragraph_ID","start_token_within_sentence","end_token_within_sentence"]] = mention_df[["paragraph_ID","start_token_within_sentence","end_token_within_sentence"]].astype(int)
    
    # Filter coreference links
    coref_df = ann_df[ann_df[0]=="COREF"].reset_index(drop=True)
    coref_df = coref_df[[1,2]]
    coref_df.columns = ["mention_ID","COREF_name"]
    
    # Merge mentions with coreference info
    entities_df = pd.merge(coref_df, mention_df, on="mention_ID", how="outer")\
                     .drop(columns=["mention_ID"])\
                     .sort_values(["paragraph_ID","start_token_within_sentence","end_token_within_sentence"])\
                     .reset_index(drop=True)
    
    # Fill missing COREF_name with unique names
    existing = set(entities_df["COREF_name"].dropna())
    for idx, row in entities_df[entities_df["COREF_name"].isna()].iterrows():
        base = row["text"].replace(" ", "_")
        i = 1
        name = base
        while name in existing:
            i += 1
            name = f"{base}_{i}"
        entities_df.at[idx, "COREF_name"] = name
        existing.add(name)

    ```

We split the original text into paragraphs and tokens, compute tokens byte offsets, and directly merge them with the mentions to map token spans to text offsets.

??? Abstract "Python Code"
    ```python
    
    # Split text into tokens and track byte offsets
    tokens = []
    reconstructed_text = ""
    for paragraph_ID, paragraph in enumerate(text_content.split("\n")):
        if paragraph_ID != 0:
            reconstructed_text += "\n"
        for token_ID_within_sentence, token in enumerate(paragraph.split(" ")):
            byte_onset = len(reconstructed_text)
            reconstructed_text += token
            byte_offset = len(reconstructed_text)
            if token_ID_within_sentence != len(paragraph.split(" ")) - 1:
                reconstructed_text += " "
            tokens.append({
                "paragraph_ID": paragraph_ID,
                "token_ID_within_sentence":token_ID_within_sentence,
                "byte_onset": byte_onset,
                "byte_offset": byte_offset,
            })
    
    tokens_df = pd.DataFrame(tokens)
    
    # Map mention token spans to byte offsets
    df = pd.merge(entities_df,
                  tokens_df[["paragraph_ID", "token_ID_within_sentence", "byte_onset"]],
                  left_on=["paragraph_ID", "start_token_within_sentence"],
                  right_on=["paragraph_ID", "token_ID_within_sentence"],
                  ).drop(columns=["token_ID_within_sentence"])
    df = pd.merge(df,
                  tokens_df[["paragraph_ID", "token_ID_within_sentence","byte_offset"]],
                  left_on=["paragraph_ID", "end_token_within_sentence"],
                  right_on=["paragraph_ID", "token_ID_within_sentence"],
                  ).drop(columns=["token_ID_within_sentence"])
    entities_df = df.copy()
    
    ```
We now have two key objects for downstream coreference tasks:

  - `reconstructed_text`: the original text stripped of trailing spaces and formatted with simple `\n` between paragraphs.
  - `entities_df`: a [`pandas.dataframe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) containing all annotated mentions with byte offsets aligned to reconstructed_text and associated coreference information.  

The **minimal configuration** of `entities_df` needed to train a coreference resolution pipeline is:

  - `byte_onset`: exact character index of the mention start in `reconstructed_text`,
  - `byte_offset`: exact character index of the mention end in `reconstructed_text`,
  - `cat`: the mention type (e.g. `PER`, `FAC`, `GPE`, `ORG`),
  - `COREF_name`: a unique identifier for the mention coreference chain.  
    
In this case `entities_df` contains additional columns:

  - `text`: the surface form of the mention,
  - `prop`: the manually annotated grammatical type of the mention (proper, noun phrase, pronoun).

| **byte_onset**   | **byte_offset**   | **cat**   | **COREF_name**            | text                | prop   |
|:-----------------|:------------------|:----------|:--------------------------|:--------------------|:-------|
| **13**           | **21**            | **FAC**   | **Chancery-0**            | Chancery            | PROP   |
| **22**           | **28**            | **GPE**   | **London-1**              | London              | PROP   |
| **69**           | **84**            | **PER**   | **Lord_Chancellor-2**     | Lord Chancellor     | PROP   |
| **96**           | **115**           | **FAC**   | **Lincoln__s_Inn_Hall-3** | Lincoln 's Inn Hall | PROP   |
| **163**          | **174**           | **FAC**   | **the_streets-4**         | the streets         | NOM    |

We can now save our formatted `reconstructed_text` and `entities_df` to our dataset directory:

??? Abstract "Python Code"
    ```python
    
    from propp_fr import save_text_file, save_entities_df

    local_dataset_path = "datasets/litbank"
    os.makedirs(local_dataset_path, exist_ok=True)
    
    save_text_file(reconstructed_text, file_name, local_dataset_path)
    save_entities_df(entities_df, file_name, local_dataset_path)

    ```

Finally, we can wrap the entire preprocessing pipeline into a loop to handle all 100 LitBank files.

??? Abstract "Python Code"
    ```python
    
    from tqdm.auto import tqdm
    import pandas as pd
    from pathlib import Path
    import os
    from propp_fr import save_text_file, save_entities_df
    
    raw_files_directory_path = "datasets/litbank/tsv"
    local_dataset_path = "datasets/litbank/propp_minimal"
    
    file_names = sorted(list(set([p.stem for p in Path(raw_files_directory_path).iterdir() if p.is_file()])))
    
    for file_name in tqdm(file_names):
        # Loading Input Files
        with open(os.path.join(raw_files_directory_path, file_name + ".txt"), "r", encoding="utf-8") as f:
            text_content = f.read()
        ann_df = pd.read_csv(os.path.join(raw_files_directory_path, file_name + ".ann"), sep="\t", header=None)
    
        # Filter mentions
        mention_df = ann_df[ann_df[0]=="MENTION"].reset_index(drop=True)
        mention_df = mention_df[[1,2,3,5,6,7,8]]
        mention_df.columns = ["mention_ID","paragraph_ID","start_token_within_sentence","end_token_within_sentence","text","cat","prop"]
        mention_df[["paragraph_ID","start_token_within_sentence","end_token_within_sentence"]] = mention_df[["paragraph_ID","start_token_within_sentence","end_token_within_sentence"]].astype(int)
    
        # Filter coreference links
        coref_df = ann_df[ann_df[0]=="COREF"].reset_index(drop=True)
        coref_df = coref_df[[1,2]]
        coref_df.columns = ["mention_ID","COREF_name"]
    
        # Merge mentions with coreference info
        entities_df = pd.merge(coref_df, mention_df, on="mention_ID", how="outer")\
                         .drop(columns=["mention_ID"])\
                         .sort_values(["paragraph_ID","start_token_within_sentence","end_token_within_sentence"])\
                         .reset_index(drop=True)
    
        # Fill missing COREF_name with unique names
        existing = set(entities_df["COREF_name"].dropna())
        for idx, row in entities_df[entities_df["COREF_name"].isna()].iterrows():
            base = row["text"].replace(" ", "_")
            i = 1
            name = base
            while name in existing:
                i += 1
                name = f"{base}_{i}"
            entities_df.at[idx, "COREF_name"] = name
            existing.add(name)
    
        # Split text into tokens and track byte offsets
        tokens = []
        reconstructed_text = ""
        for paragraph_ID, paragraph in enumerate(text_content.split("\n")):
            if paragraph_ID != 0:
                reconstructed_text += "\n"
            for token_ID_within_sentence, token in enumerate(paragraph.split(" ")):
                byte_onset = len(reconstructed_text)
                reconstructed_text += token
                byte_offset = len(reconstructed_text)
                if token_ID_within_sentence != len(paragraph.split(" ")) - 1:
                    reconstructed_text += " "
                tokens.append({
                    "paragraph_ID": paragraph_ID,
                    "token_ID_within_sentence":token_ID_within_sentence,
                    "byte_onset": byte_onset,
                    "byte_offset": byte_offset,
                })
    
        tokens_df = pd.DataFrame(tokens)
    
        # Map mention token spans to byte offsets
        df = pd.merge(entities_df,
                      tokens_df[["paragraph_ID", "token_ID_within_sentence", "byte_onset"]],
                      left_on=["paragraph_ID", "start_token_within_sentence"],
                      right_on=["paragraph_ID", "token_ID_within_sentence"],
                      ).drop(columns=["token_ID_within_sentence"])
        df = pd.merge(df,
                      tokens_df[["paragraph_ID", "token_ID_within_sentence","byte_offset"]],
                      left_on=["paragraph_ID", "end_token_within_sentence"],
                      right_on=["paragraph_ID", "token_ID_within_sentence"],
                      ).drop(columns=["token_ID_within_sentence", "paragraph_ID", "start_token_within_sentence", "end_token_within_sentence"])
        entities_df = df.copy()
    
        minimal_columns = ['byte_onset', 'byte_offset', 'cat', 'COREF_name']
        entities_df = entities_df[minimal_columns + [col for col in df.columns if col not in minimal_columns]]
    
        file_name = file_name.replace("_brat", "") if file_name.endswith("_brat") else file_name
        save_text_file(reconstructed_text, file_name, local_dataset_path)
        save_entities_df(entities_df, file_name, local_dataset_path)
    
    import requests
    
    split_id = 0
    split_config = {}
    
    while True:
        url = f"https://raw.githubusercontent.com/dbamman/lrec2020-coref/master/data/litbank_tenfold_splits/{split_id}/test.ids"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            # No more splits (404 or network error)
            break
    
        content = response.text
    
        split_content = [
            file_name.replace("_brat.tsv", "")
            for file_name in content.split("\n")
            if file_name.endswith("_brat.tsv")
        ]
    
        split_config[f"test_{split_id}"] = split_content
        split_id += 1
    
    import json
    json.dump(split_config, open(os.path.join(local_dataset_path, "split_config.json"), "w"))
    
    import shutil
    
    output_archive_name = "litbank_propp_minimal_implementation"
    # Path where the ZIP will be saved (same level as local_dataset_path)
    output_path = os.path.join(os.path.dirname(local_dataset_path), output_archive_name)
    # Create a ZIP archive
    shutil.make_archive(output_path, 'zip', root_dir=local_dataset_path)
    print(f"Archive created: {output_path}.zip")

    ```

Finally, the script generates a **compressed ZIP archive** of the processed dataset:

```
litbank_propp_minimal_implementation.zip
```

This archive is ready to use with the Propp pipeline.

Alternatively, the dataset archive can be downloaded directly from the [datasets section](datasets).





