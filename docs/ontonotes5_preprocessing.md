# OntoNotes 5 Preprocessing

Tutorial on how to preprocess the [OntoNotes 5.0 Dataset](https://catalog.ldc.upenn.edu/LDC2013T19) for Named-Entity Recognition (NER).

This Dataset is subject to law and can't be shared freely, we will not share the dataset neither raw or after preprocessing, this tutorial can be used by anyone that obtained access to the dataset by Linguistic Data Consortium.
It is free. It is easy you just need to follow the tutorial.
[Licensing Instructions: 	Subscription & Standard Members, and Non-Members](https://www.ldc.upenn.edu/language-resources/data/obtaining)

Start with a directory containing the `ontonotes-release-5.0_LDC2013T19.tgz` archive.


??? Abstract "**Python Code to Preprocess the CoNLL-2003 NER Dataset**"
    ```python
            
    #%%
    import os
    import urllib.request
    import zipfile
    
    import pandas as pd
    from collections import Counter
    from tqdm.auto import tqdm
    
    from propp_fr import save_text_file, save_entities_df
    
    # URL of the dataset
    url = "https://data.deepai.org/conll2003.zip"
    
    # Destination paths
    data_dir = "datasets/conll2003"
    zip_path = os.path.join(data_dir, "conll2003.zip")
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the zip file
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete!")
    
    # Unzip
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete!")
    #%%
    def generate_split_tokens_df(split_content):
        doc_ID = -1
        paragraph_ID = 0
    
        tokens_df = []
    
        for row in split_content:
            row = row.strip()
            if row == '-DOCSTART- -X- -X- O':
                doc_ID += 1
                token_ID = 0
                paragraph_ID = -1
            elif row == "":
                paragraph_ID += 1
            elif row != "":
                word, POS, syntactic, BIO = row.split(" ")
                tokens_df.append({"token_ID":token_ID,
                                  "doc_ID":doc_ID,
                                  "paragraph_ID":paragraph_ID,
                                  "word":word,
                                  "POS":POS,
                                  "syntactic":syntactic,
                                  "BIO":BIO})
                token_ID += 1
    
        tokens_df = pd.DataFrame(tokens_df)
        return tokens_df
    
    def extract_conll2003_entities_df(tokens_df):
        entities = []
    
        entity = None
    
        for token_ID, BIO in enumerate(tokens_df["BIO"]):
    
            if BIO == "O":
                if entity is not None:
                    entity["end"] = token_ID - 1
                    entity["len"] = entity["end"] - entity["start"] + 1
                    entities.append(entity)
                    entity = None
                continue
    
            tag, cat = BIO.split("-", 1)
    
            if tag == "B":
                if entity is not None:
                    entity["end"] = token_ID - 1
                    entity["len"] = entity["end"] - entity["start"] + 1
                    entities.append(entity)
                entity = {"cat": cat, "start": token_ID}
    
            elif tag == "I":
                if entity is None or entity["cat"] != cat:
                    # illegal I after O or type mismatch → treat as B
                    if entity is not None:
                        entity["end"] = token_ID - 1
                        entity["len"] = entity["end"] - entity["start"] + 1
                        entities.append(entity)
                    entity = {"cat": cat, "start": token_ID}
                # else: valid continuation → do nothing
    
        # close entity at end of document
        if entity is not None:
            entity["end"] = len(tokens_df) - 1
            entity["len"] = entity["end"] - entity["start"] + 1
            entities.append(entity)
    
        entities_df = pd.DataFrame(entities)
        return entities_df
    
    def align_entities_byte_offsets(tokens_df, entities_df):
        tokens = []
        reconstructed_text = ""
        token_ID = 0
        for paragraph_ID, paragraph_tokens_df in tokens_df.groupby("paragraph_ID"):
            for token_ID_within_sentence, token in enumerate(paragraph_tokens_df["word"].tolist()):
                byte_onset = len(reconstructed_text)
                reconstructed_text += str(token)
                byte_offset = len(reconstructed_text)
                if token_ID_within_sentence != len(paragraph_tokens_df) - 1:
                    reconstructed_text += " "
                tokens.append({
                    "paragraph_ID": paragraph_ID,
                    "token_ID_within_sentence":token_ID_within_sentence,
                    "token_ID":token_ID,
                    "byte_onset": byte_onset,
                    "byte_offset": byte_offset,
                })
                token_ID += 1
            if paragraph_ID != len(tokens_df["paragraph_ID"].unique()) - 1:
                reconstructed_text += "\n"
    
        tokens_df = pd.DataFrame(tokens)
        # print(tokens)
        # Map mention token spans to byte offsets
        df = pd.merge(entities_df,
                      tokens_df[["token_ID", "byte_onset"]],
                      left_on="start",
                      right_on="token_ID",
                      how="left"
                      ).drop(columns=["token_ID"])
    
        df = pd.merge(df,
                      tokens_df[["token_ID", "byte_offset"]],
                      left_on="end",
                      right_on="token_ID",
                      how="left"
                      ).drop(columns=["token_ID"])
        entities_df = df.copy()
        return reconstructed_text, entities_df, tokens_df
    
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
    #%%
    import json
    
    files_directory = "datasets/conll2003/conll2003-NER_propp_minimal_implementation"
    
    split_config = {}
    
    for split_name in ["train", "valid", "test"]:
        split_config[split_name] = []
        print(f"Split: {split_name}")
        with open(os.path.join(data_dir, f"{split_name}.txt"), "r") as f:
            split_content = f.readlines()
        tokens_df = generate_split_tokens_df(split_content)
    
    
        for doc_id, doc_tokens_df in tqdm(tokens_df.groupby("doc_ID")):
            doc_entities_df = extract_conll2003_entities_df(doc_tokens_df)
            reconstructed_text, doc_entities_df, doc_tokens_df = align_entities_byte_offsets(doc_tokens_df, doc_entities_df)
            doc_entities_df = doc_entities_df[["cat", "byte_onset", "byte_offset"]]
    
            file_name = f"{split_name}_{doc_id}"
            save_text_file(reconstructed_text, file_name, files_directory)
            save_entities_df(doc_entities_df, file_name, files_directory)
    
            split_config[split_name].append(file_name)
    
    json.dump(split_config, open(os.path.join(files_directory, "split_config.json"), "w"))
    
    import shutil
    # Create a ZIP archive
    shutil.make_archive(files_directory, 'zip', root_dir=files_directory)
    print(f"Archive created: {files_directory}.zip")
    #%%

    ```


This archive is ready to use with the Propp pipeline.

Alternatively, the dataset archive can be downloaded directly from the [datasets section](datasets).