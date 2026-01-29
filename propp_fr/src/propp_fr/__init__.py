import pandas as pd

import sys
from tqdm.auto import tqdm

import propp_fr
# Patch sys.modules to redirect references to 'propp_fr' to 'propp_fr'
sys.modules['propp_fr'] = propp_fr

from .propp_fr_load_save_functions import load_sacr_file, load_text_file, save_text_file, load_tokens_df, save_tokens_df, load_entities_df, save_entities_df, clean_text, load_book_file, save_book_file
from .propp_fr_add_entities_features import add_features_to_entities
from .propp_fr_generate_tokens_df import load_spacy_model, generate_tokens_df
from .propp_fr_generate_tokens_and_entities_from_sacr import generate_tokens_and_entities_from_sacr
from .propp_fr_generate_tokens_embeddings_tensor import load_tokenizer_and_embedding_model, get_embedding_tensor_from_tokens_df
from .propp_fr_mentions_detection_module import mentions_detection_LOOCV_full_model_training, generate_NER_model_card_from_LOOCV_directory, load_mentions_detection_model, generate_entities_df
from .propp_fr_mentions_detection_module import LockedDropout, Highway, NERModel

from .propp_fr_coreference_resolution_module import coreference_resolution_LOOCV_full_model_training, generate_coref_model_card_from_LOOCV_directory
from .propp_fr_coreference_resolution_module import load_coreference_resolution_model, perform_coreference, CoreferenceResolutionModel

from .propp_fr_extract_attributes import extract_attributes
from .propp_fr_generate_characters_dict import generate_characters_dict

from .propp_fr_generate_sacr_file import generate_sacr_file

from .propp_fr_single_line_command import process_text_file, load_models

from .propp_fr_generate_character_network import generate_character_network

# Inside propp_fr/__init__.py
print("propp_fr package loaded successfully.")




