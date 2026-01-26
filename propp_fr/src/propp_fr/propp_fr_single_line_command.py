import os
import spacy
import torch
import traceback
import subprocess
import sys
import json

from .propp_fr_load_save_functions import load_text_file, clean_text, save_text_file, save_tokens_df, save_entities_df, save_book_file
from .propp_fr_generate_tokens_df import load_spacy_model, generate_tokens_df
from .propp_fr_mentions_detection_module import load_mentions_detection_model, generate_entities_df
from .propp_fr_coreference_resolution_module import load_coreference_resolution_model,perform_coreference
from .propp_fr_generate_tokens_embeddings_tensor import get_embedding_tensor_from_tokens_df, load_tokenizer_and_embedding_model
from .propp_fr_add_entities_features import add_features_to_entities
from .propp_fr_extract_attributes import extract_attributes
from .propp_fr_generate_characters_dict import generate_characters_dict

def install_spacy_for_cuda():
    #print("Checking SpaCy installation...")
    
    # Determine the correct package
    cuda_version = "cpu"
    if torch.version.cuda:
        cuda_version = torch.version.cuda.split(".")[0] + "x"
    package = f"spacy[cuda{cuda_version}]" if cuda_version != "cpu" else "spacy"
    
    # Check if spacy is already installed
    try:
        import spacy
        #print(f"SpaCy {spacy.__version__} is already installed.")
        
        # Check if GPU is available and matches the desired CUDA version
        if cuda_version != "cpu":
            try:
                spacy.prefer_gpu() #spacy.require_gpu()
                #print(f"SpaCy is already configured to use GPU with CUDA {cuda_version}. Skipping installation.")
                return
            except Exception:
                print("GPU support is missing or incompatible. Reinstalling...")
        else:
            print("CPU version detected. No GPU required. Skipping installation.")
            return
    except ImportError:
        print("SpaCy not found. Installing...")

    # Install or upgrade
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package], check=True)
    print("SpaCy installation complete.")



def load_models(spacy_model_name="fr_dep_news_trf",
                mentions_detection_model_name="AntoineBourgois/propp-fr_NER_camembert-large_FAC_GPE_LOC_PER_TIME_VEH",
                coreference_resolution_model_name="AntoineBourgois/propp-fr_coreference-resolution_camembert-large_PER",
                spacy_input_max_length=500000,
                force_download=False): \
        # Load models once
    print("Loading models...")
    install_spacy_for_cuda()
    try:
        if torch.cuda.is_available():
            spacy.require_gpu()  # prefer_gpu()
            spacy_model = spacy.load(spacy_model_name)
            print("CUDA is required, Spacy model should run on GPU.")
        else:
            spacy_model = load_spacy_model(model_name=spacy_model_name, model_max_length=spacy_input_max_length)
            print("CUDA is not available, Spacy model will run on CPU.")
    except OSError:
        subprocess.run([sys.executable, '-m', 'spacy', 'download', spacy_model_name], check=True)
        if torch.cuda.is_available():
            spacy.prefer_gpu()  # require_gpu()
            spacy_model = spacy.load(spacy_model_name)
            print("CUDA is preferred, Spacy model should run on GPU.")
        else:
            spacy_model = load_spacy_model(model_name=spacy_model_name, model_max_length=spacy_input_max_length)
            print("CUDA is not available, Spacy model will run on CPU.")
    
    
    
    spacy_model.max_length = spacy_input_max_length

    mentions_detection_model = load_mentions_detection_model(
        model_path=mentions_detection_model_name,
        force_download=force_download
    )
    coreference_resolution_model = load_coreference_resolution_model(
        model_path=coreference_resolution_model_name,
        force_download=force_download
    )

    if mentions_detection_model["base_model_name"] != coreference_resolution_model["base_model_name"]:
        print(
            f"WARNING: mentions_detection_model and coreference_resolution_model DO NOT use the same embedding model.")
    
    print(f"\nModels Loaded Successfully:\nSpacy: {spacy_model_name}\nMentions Detection: {mentions_detection_model_name}\nCoreference Resolution: {coreference_resolution_model_name}")

    return spacy_model, mentions_detection_model, coreference_resolution_model


def process_file(file_name,
                 input_folder,
                 output_folder,
                 spacy_model,
                 mentions_detection_model,
                 coreference_resolution_model,
                 tokenizer, embedding_model,
                 spacy_max_characters_batch=1000000,
                 embedding_mini_batch=64,
                 mentions_detection_batch=14,
                 coreference_resolution_batch=50000,
                 verbose=1
                 ):
    """
    Process a single text and save outputs to the specified folder.
    """

    # Process the file
    txt_content = load_text_file(file_name, files_directory=input_folder)
    txt_content = clean_text(txt_content)
    tokens_df = generate_tokens_df(txt_content,
                                   spacy_model,
                                   max_char_sentence_length=spacy_max_characters_batch
                                   )

    # Generate token embeddings
    if verbose: print("Generating token embeddings...")
    tokens_embedding_tensor = get_embedding_tensor_from_tokens_df(txt_content,
                                                                  tokens_df,
                                                                  tokenizer,
                                                                  embedding_model,
                                                                  sliding_window_size='max',
                                                                  mini_batch_size=embedding_mini_batch,
                                                                  sliding_window_overlap=0.5,
                                                                  subword_pooling_strategy=mentions_detection_model[
                                                                      "subword_pooling_strategy"])

    if verbose: print("Mentions Detection...")
    entities_df = generate_entities_df(tokens_df,
                                       tokens_embedding_tensor,
                                       mentions_detection_model,
                                       batch_size=mentions_detection_batch
                                       )

    entities_df = add_features_to_entities(entities_df, tokens_df)

    # Perform Coreference Resolution
    if coreference_resolution_model["base_model_name"] != mentions_detection_model["base_model_name"]:
        tokenizer, embedding_model = load_tokenizer_and_embedding_model(coreference_resolution_model["base_model_name"])
        tokens_embedding_tensor = get_embedding_tensor_from_tokens_df(txt_content,
                                                                      tokens_df,
                                                                      tokenizer,
                                                                      embedding_model,
                                                                      sliding_window_size='max',
                                                                      mini_batch_size=embedding_mini_batch,
                                                                      sliding_window_overlap=0.5,
                                                                      subword_pooling_strategy=mentions_detection_model[
                                                                          "subword_pooling_strategy"])

    if verbose: print("Coreference Resolution...")
    entities_df = perform_coreference(entities_df=entities_df,
                                      tokens_embedding_tensor=tokens_embedding_tensor,
                                      coreference_resolution_model=coreference_resolution_model,
                                      batch_size=coreference_resolution_batch,
                                      propagate_coref=True,
                                      rule_based_postprocess=False)

    # Extract character attributes
    tokens_df = extract_attributes(entities_df, tokens_df)
    characters_dict = generate_characters_dict(tokens_df, entities_df)

    # Save all files
    save_text_file(txt_content, file_name, files_directory=output_folder)
    save_tokens_df(tokens_df, file_name, files_directory=output_folder)
    save_entities_df(entities_df, file_name, files_directory=output_folder)

    save_book_file(characters_dict, file_name, files_directory=output_folder)


def process_text_file(txt_file_path):
    if not os.path.isfile(txt_file_path):
        print(f"File {txt_file_path} does not exist.")

    else:
        spacy_model_name = "fr_dep_news_trf"
        spacy_input_max_length = 500000

        print("Loading models...")
        if torch.cuda.is_available():
            spacy.prefer_gpu()  # prefer
            spacy_model = spacy.load(spacy_model_name)
            print("CUDA is available, Spacy model should run on GPU.")
        else:
            spacy_model = load_spacy_model(model_name=spacy_model_name, model_max_length=spacy_input_max_length)
            print("CUDA is not available, Spacy model will run on CPU.")
        spacy_model.max_length = spacy_input_max_length

        # Split into directory and file name
        input_folder = os.path.dirname(txt_file_path)
        file_name = os.path.basename(txt_file_path)
        file_name, _ = os.path.splitext(file_name)

        output_folder = input_folder

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # 1. Loading Models
        spacy_model, mentions_detection_model, coreference_resolution_model = load_models(
            spacy_model_name="fr_dep_news_trf",
            mentions_detection_model_name = "AntoineBourgois/propp-fr_NER_camembert-large_FAC_GPE_LOC_PER_TIME_VEH",
            # mentions_detection_model_name="AntoineBourgois/propp-fr_NER_camembert-large_PER",
            coreference_resolution_model_name="AntoineBourgois/propp-fr_coreference-resolution_camembert-large_PER",
            force_download=False)
        tokenizer, embedding_model = load_tokenizer_and_embedding_model(mentions_detection_model["base_model_name"])

        if os.path.exists(os.path.join(output_folder, f"{file_name}.book")):
            print(f"{file_name}.book already exists. Skipping...")

        else:
            process_file(file_name,
                         input_folder,
                         output_folder,
                         spacy_model,
                         mentions_detection_model,
                         coreference_resolution_model,
                         tokenizer, embedding_model,
                         verbose=1,
                         spacy_max_characters_batch=50000,
                         embedding_mini_batch=10,
                         mentions_detection_batch=12,
                         coreference_resolution_batch=50000,

                         # spacy_max_characters_batch=1000000,
                         # embedding_mini_batch=64,
                         # mentions_detection_batch=128,
                         # coreference_resolution_batch=50000,

                         )

            try:
                process_file(file_name,
                             input_folder,
                             output_folder,
                             spacy_model,
                             mentions_detection_model,
                             coreference_resolution_model,
                             tokenizer, embedding_model,
                             verbose=1,
                             spacy_max_characters_batch=50000,
                             embedding_mini_batch=10,
                             mentions_detection_batch=12,
                             coreference_resolution_batch=50000,

                             # spacy_max_characters_batch=1000000,
                             # embedding_mini_batch=64,
                             # mentions_detection_batch=128,
                             # coreference_resolution_batch=50000,

                             )
            except Exception as e:
                print(f"⚠️ Unexpected error processing file {file_name}: {e}")
                traceback.print_exc()  # <- prints the full traceback
