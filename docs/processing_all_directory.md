---
title: "Processing Full Directory"
---

# Processing all .txt Files in a Directory



??? Abstract "**You can copy / paste the whole Notebook Code**"

    ```python
    from propp_fr import load_models, load_text_file, generate_tokens_df, load_tokenizer_and_embedding_model, get_embedding_tensor_from_tokens_df, generate_entities_df, add_features_to_entities, perform_coreference, extract_attributes, generate_characters_dict, save_tokens_df, save_entities_df, save_book_file
    
    from pathlib import Path
    from tqdm.auto import tqdm
    
    files_directory = "<directory_containing_txt_files>"

    txt_files = sorted(p.stem for p in Path(files_directory).iterdir() if p.suffix == ".txt")
    book_files = sorted(p.stem for p in Path(files_directory).iterdir() if p.suffix == ".book")
    
    unprocessed_files = [file for file in txt_files if file not in book_files]
    print(f"Unprocessed Files: {len(unprocessed_files):,}")
    
    spacy_model, mentions_detection_model, coreference_resolution_model = load_models()
    tokenizer, embedding_model = load_tokenizer_and_embedding_model(mentions_detection_model["base_model_name"])
    
    for file_name in tqdm(unprocessed_files, desc="Processing .txt Files"):
        print(f"Processing: {file_name}...")
        text_content = load_text_file(file_name, files_directory)
        tokens_df = generate_tokens_df(text_content, spacy_model)
        tokens_embedding_tensor = get_embedding_tensor_from_tokens_df(
            text_content,
            tokens_df,
            tokenizer,
            embedding_model
        )
    
        entities_df = generate_entities_df(
            tokens_df,
            tokens_embedding_tensor,
            mentions_detection_model,
        )
    
        entities_df = add_features_to_entities(entities_df, tokens_df)
    
        entities_df = perform_coreference(
            entities_df,
            tokens_embedding_tensor,
            coreference_resolution_model,
            propagate_coref=True,
            rule_based_postprocess=False,
        )
    
        tokens_df = extract_attributes(entities_df, tokens_df)
    
        characters_dict = generate_characters_dict(tokens_df, entities_df)
    
    
        save_tokens_df(tokens_df, file_name, files_directory)
        save_entities_df(entities_df, file_name, files_directory)
        save_book_file(characters_dict, file_name, files_directory)
    ```
