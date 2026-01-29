import sys
import warnings
import os
import pandas as pd
from itertools import product
from itertools import combinations
from transformers import AutoConfig

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(
    "ignore",
    message="^To copy construct from a tensor",
    category=UserWarning
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torch.nn as nn

import numpy as np
import random
from sklearn.metrics import classification_report
import gc
import re
from itertools import combinations
import json
import subprocess
from tabulate import tabulate
import requests

from propp_fr import load_tokenizer_and_embedding_model, get_embedding_tensor_from_tokens_df
from propp_fr import load_tokens_df, load_entities_df, load_text_file

def get_tokens_embeddings_tensor_dict(tokens_embeddings_tensor_dict_path, model_name, files_directory,
                                      embedding_batch_size=10, subword_pooling_strategy="average",
                                      verbose=1):
    """
    Generate or load a dictionary containing contextual embeddings for each token in the documents.

    Args:
        tokens_embeddings_tensor_dict_path (str): Path to save/load the embeddings dictionary.
        model_name (str): Name of the embedding model to use.
        files_directory (str): Directory containing token files.
        embedding_batch_size (int, optional): Batch size for embedding generation. Default is 10.

    Returns:
        dict: A dictionary where keys are file names and values are token embedding tensors.
    """
    # Check if the embeddings dictionary already exists
    if not os.path.exists(tokens_embeddings_tensor_dict_path):
        # Load tokenizer and embedding model
        tokenizer, embedding_model = load_tokenizer_and_embedding_model(model_name)

        # Get all .tokens files in the directory
        extension = ".tokens"
        tokens_files = sorted([f.replace(extension, "") for f in os.listdir(files_directory) if f.endswith(extension)])

        # Initialize dictionary to store embeddings
        tokens_embeddings_tensor_dict = {}

        # Generate embeddings for each file
        if verbose == 1:
            for file_name in tqdm(tokens_files, desc="Generating tokens embeddings", leave=False):
                tokens_df = load_tokens_df(file_name, files_directory=files_directory, extension=extension)
                text_content = load_text_file(file_name, files_directory=files_directory)

                tokens_embedding_tensor = get_embedding_tensor_from_tokens_df(text_content, tokens_df,
                                                                              tokenizer, embedding_model,
                                                                              sliding_window_size='max',
                                                                              mini_batch_size=embedding_batch_size,
                                                                              sliding_window_overlap=0.5,
                                                                              subword_pooling_strategy=subword_pooling_strategy,
                                                                              )

                # Store embeddings in the dictionary
                tokens_embeddings_tensor_dict[file_name] = tokens_embedding_tensor
        if verbose == 0:
            for file_name in tokens_files:
                tokens_df = load_tokens_df(file_name, files_directory=files_directory, extension=extension)
                text_content = load_text_file(file_name, files_directory=files_directory)

                tokens_embedding_tensor = get_embedding_tensor_from_tokens_df(text_content, tokens_df,
                                                                              tokenizer, embedding_model,
                                                                              sliding_window_size='max',
                                                                              mini_batch_size=embedding_batch_size,
                                                                              sliding_window_overlap=0.5,
                                                                              subword_pooling_strategy=subword_pooling_strategy,
                                                                              )

                # Store embeddings in the dictionary
                tokens_embeddings_tensor_dict[file_name] = tokens_embedding_tensor

        # Save the generated embeddings dictionary to disk
        with open(tokens_embeddings_tensor_dict_path, 'wb') as f:
            pickle.dump(tokens_embeddings_tensor_dict, f)

    else:  # if the tokens embeddings dict already exist
        # Load the embeddings dictionary from disk
        with open(tokens_embeddings_tensor_dict_path, 'rb') as f:
            tokens_embeddings_tensor_dict = pickle.load(f)

    return tokens_embeddings_tensor_dict

def get_mentions_embeddings(entities_df, tokens_embeddings_tensor):
    """
    Generate a single embedding for each mention by averaging the first and last token embeddings.

    Args:
        entities_df (pd.DataFrame): DataFrame containing mention start and end token indices.
            Must include 'start_token' and 'end_token' columns.
        tokens_embeddings_tensor (torch.Tensor): Tensor containing token embeddings,
            where the index corresponds to token positions in the document.

    Returns:
        torch.Tensor: A tensor where each row is the averaged embedding of a mention.
    """
    mentions_embeddings = []

    for entity_start_token, entity_end_token in entities_df[['start_token', 'end_token']].values:
        # Extract the first and last token embeddings
        first_last_embeddings = [tokens_embeddings_tensor[entity_start_token],
                                 tokens_embeddings_tensor[entity_end_token]]

        # Compute the mean of the first and last embeddings
        mention_embedding = torch.mean(torch.stack(first_last_embeddings), dim=0)

        # Instead concatenate first and last embeddings
        # mention_embedding = torch.cat(first_last_embeddings, dim=0)

        # Append the computed embedding
        mentions_embeddings.append(mention_embedding)

    if mentions_embeddings:
        mentions_embeddings_tensor = torch.stack(mentions_embeddings)
    else:
        # Return an empty tensor if no embeddings were successfully computed
        mentions_embeddings_tensor = torch.empty(0)

    return mentions_embeddings_tensor

def initialize_mention_pairs_df(
        entities_df,
        pronoun_antecedent_max_distance=30,
        proper_common_nouns_antecedent_max_distance=300,
        low_information_noun_max_distance=50,
):
    """
    Initialize a DataFrame of valid mention pairs within a specified maximum distance.
    The maximum distance depends on the 'prop' category of mentions (e.g., 'PROP', 'NOM', 'PRON').

    Args:
        entities_df (pd.DataFrame): DataFrame containing mention information.
            Must include 'prop' (mention category) and 'mention_len' (mention length) columns.
        pronoun_antecedent_max_distance (int): Maximum distance allowed for pronoun antecedents (default: 30).
        proper_common_nouns_antecedent_max_distance (int): Maximum distance for proper/common nouns (default: 300).
        low_information_noun_max_distance (int): Maximum distance for low-information nouns (default: 50).

    Returns:
        pd.DataFrame: A DataFrame with columns "A" and "B", where:
            - "A" is the antecedent mention index.
            - "B" is the current mention index.
    """
    # Convert columns to NumPy arrays for faster access
    mentions_prop_list = entities_df["prop"].to_numpy()
    mentions_len_list = entities_df["mention_len"].to_numpy()
    N = len(mentions_prop_list)

    mention_pairs = []

    # Iterate over each mention as the current mention (B)
    for B in range(N):
        # Determine the max distance for antecedents (A) based on the current mention (B)
        if "PRON" in mentions_prop_list[B]:
            max_distance = pronoun_antecedent_max_distance
        elif "NOM" in mentions_prop_list[B] and mentions_len_list[B] == 1:
            max_distance = low_information_noun_max_distance
        else:
            max_distance = proper_common_nouns_antecedent_max_distance

        # Identify valid antecedents (A)
        for A in range(max(0, B - max_distance), B):
            mention_pairs.append((A, B))

    # Create the mention pairs DataFrame
    mention_pairs_df = pd.DataFrame(mention_pairs, columns=["A", "B"])

    return mention_pairs_df


# Generate Features array -------------------------------------------------------------
def get_mention_pairs_distance_features(mention_pairs_df, features=None):

    """
    Calculate distance-based features for mention pairs.

    Args:
        mention_pairs_df (pd.DataFrame): DataFrame containing mention pair indices with columns "A" and "B".
        features (list of str, optional): List of distance-based features to calculate (default includes various deltas).

    Returns:
        pd.DataFrame: Enriched mention_pairs_df with distance-based features added as columns.
    """
    if features==None:
        features=['mention_ID_delta', 'start_token_delta', 'end_token_delta', 'paragraph_ID_delta', 'sentence_ID_delta', 'out_to_in_nested_level_delta']

    # Define mapping of feature names to efficient vectorized calculations
    feature_mappings = {
        'mention_ID_delta': lambda df: abs(df['A'] - df['B']),
        'start_token_delta': lambda df: df['B_start_token'] - df['A_start_token'],
        'end_token_delta': lambda df: df['B_end_token'] - df['A_end_token'],
        'paragraph_ID_delta': lambda df: abs(df['A_paragraph_ID'] - df['B_paragraph_ID']),
        'sentence_ID_delta': lambda df: abs(df['A_sentence_ID'] - df['B_sentence_ID']),
        'out_to_in_nested_level_delta': lambda df: abs(df['A_out_to_in_nested_level'] - df['B_out_to_in_nested_level']),
    }

    # Apply only selected features
    for feature in features:
        if feature in feature_mappings:
            mention_pairs_df[feature] = feature_mappings[feature](mention_pairs_df)

    return mention_pairs_df
def get_shared_token_ratio(mention_pairs_df):
    """
    Calculate the shared token ratio between two text strings.

    Args:
        A_text (str): Text of the first mention.
        B_text (str): Text of the second mention.

    Returns:
        float: Ratio of shared tokens to the length of the longer text.
    """
    shared_token_ratio_list = []
    for A_text, B_text in mention_pairs_df[['A_text', 'B_text']].values:
        A_tokens, B_tokens = str(A_text).lower().split(), str(B_text).lower().split()

        # Calculate shared tokens
        shared_tokens = set(A_tokens).intersection(set(B_tokens))
        shared_count = len(shared_tokens)

        # Calculate the length of the longer text
        longer_text_tokens_count = max(len(A_tokens), len(B_tokens))

        # Calculate the ratio
        ratio = shared_count / longer_text_tokens_count if longer_text_tokens_count > 0 else 0
        shared_token_ratio_list.append(ratio)
    return shared_token_ratio_list
def get_text_and_syntactic_match_features(mention_pairs_df, features=None):
    """
    Generate text, entity types and syntactic match features for mention pairs.

    Args:
        mention_pairs_df (pd.DataFrame): DataFrame containing mention pair indices with columns "A" and "B".
        features (list of str, optional): List of text and syntactic match features to calculate.

    Returns:
        pd.DataFrame: Enriched mention_pairs_df with the requested features added as columns.
    """
    if features == None:
        features = ['shared_token_ratio', 'text_match', 'head_text_match', 'syntactic_head_match', 'cat_match']

    if 'shared_token_ratio' in features:
        mention_pairs_df['shared_token_ratio'] = get_shared_token_ratio(mention_pairs_df)
    if 'text_match' in features:
        mention_pairs_df['text_match'] = (mention_pairs_df['A_text'] == mention_pairs_df['B_text']).astype(int)
    if 'head_text_match' in features:
        mention_pairs_df['head_text_match'] = (mention_pairs_df['A_head_word'] == mention_pairs_df['B_head_word']).astype(int)
    if 'syntactic_head_match' in features:
        mention_pairs_df['syntactic_head_match'] = (mention_pairs_df['A_head_syntactic_head_ID'] == mention_pairs_df['B_head_syntactic_head_ID']).astype(int)
    if 'cat_match' in features:
        mention_pairs_df['cat_match'] = (mention_pairs_df['A_cat'] == mention_pairs_df['B_cat']).astype(int)

    return mention_pairs_df
def get_one_hot_encoded_features(mention_pairs_df, features=None):
    """
    Generate one-hot encoded features for mention pairs.

    Args:
        mention_pairs_df (pd.DataFrame): DataFrame containing mention pair indices with columns "A" and "B".
        entities_df (pd.DataFrame): DataFrame containing mention-level categorical information.
        features (list of str, optional): List of features to one-hot encode.

    Returns:
        pd.DataFrame: Enriched mention_pairs_df with one-hot encoded features added as columns.
    """
    if features == None:
        features =['prop', 'head_dependency_relation', 'gender', 'number', 'grammatical_person']
    # Define the one-hot encoding mapping
    one_hot_encoding_dict = {"prop": ["NOM", "PROP", "PRON"],
                             "head_dependency_relation": ["ROOT", "acl", "acl:relcl", "advcl", "advmod", "amod",
                                                          "appos", "aux:pass", "aux:tense", "case", "cc", "ccomp",
                                                          "conj", "cop", "dep", "det", "expl:comp", "expl:pass",
                                                          "expl:subj", "fixed", "flat:foreign", "flat:name", "iobj",
                                                          "mark", "nmod", "nsubj", "nsubj:pass", "nummod", "obj",
                                                          "obl:agent", "obl:arg", "obl:mod", "parataxis", "punct",
                                                          "vocative", "xcomp"],
                             "gender": ["Male", "Female", "Ambiguous", 'Not_Assigned'],
                             "number": ["Singular", "Plural", "Ambiguous", 'Not_Assigned'],
                             "grammatical_person": [1, 2, 3, 4],
                             }


    columns = list(one_hot_encoding_dict.keys())
    features = [column for column in columns if column in features]

    # Perform one-hot encoding
    for mention_polarity in ["A", "B"]:
        for feature in features:
            # Generate dummy values ensuring all possible categories are included
            possible_values = one_hot_encoding_dict[feature]
            dummy_values = pd.get_dummies(mention_pairs_df[f"{mention_polarity}_{feature}"].tolist() + possible_values)[
                           :len(mention_pairs_df)].astype(int)
            # Add prefix and append to DataFrame
            dummy_values = dummy_values.add_prefix(f"{mention_polarity}_{feature}_")
            mention_pairs_df[dummy_values.columns] = dummy_values

    return mention_pairs_df

def convert_mention_pairs_df_to_features_array(mention_pairs_df, default_columns):
    """
    Converts the mention pairs DataFrame into a NumPy array of features.

    Args:
        mention_pairs_df (pd.DataFrame): DataFrame containing mention pairs with their features.

    Returns:
        np.ndarray: A NumPy array containing feature values for the mention pairs, with columns sorted.
    """
    features_columns = mention_pairs_df.drop(columns=default_columns).columns.sort_values()

    # Extract the feature columns from the DataFrame
    mention_pairs_df = mention_pairs_df[features_columns]

    # Reduce precision earlier by converting to float16
    mention_pairs_df = mention_pairs_df.astype(np.float16)

    # Convert the DataFrame to a NumPy array
    features_array = mention_pairs_df.values

    return features_array

def generate_mention_pairs_features_array(mention_pairs_df, CAT_entities_df, features=None):
    """
    Generate a feature array for mention pairs, including both pair-specific and individual mention features.

    Args:
        mention_pairs_df (pd.DataFrame): DataFrame containing mention pair indices (columns "A" and "B").
        CAT_entities_df (pd.DataFrame): DataFrame containing information about individual mentions.
        features (list of str, optional): List of feature names to include in the output array.
            Defaults to a comprehensive set of mention and mention-pair features.

    Returns:
        np.ndarray: A feature array where each row corresponds to a mention pair
        and columns represent the selected features.
    """
    if features == None:
        features=['mention_len', 'start_token_ID_within_sentence', 'mention_ID_delta',
                                                    'start_token_delta', 'end_token_delta', 'paragraph_ID_delta',
                                                    'sentence_ID_delta', 'out_to_in_nested_level_delta',
                                                    'shared_token_ratio', 'text_match', 'head_text_match',
                                                    'syntactic_head_match', 'cat_match', 'prop',
                                                    'head_dependency_relation', 'gender', 'number',
                                                    'grammatical_person']

    CAT_entities_df["text"] = CAT_entities_df["text"].fillna("").astype(str).str.lower()
    CAT_entities_df["head_word"] = CAT_entities_df["head_word"].fillna("").astype(str).str.lower()

    columns = ['cat', 'end_token', 'gender', 'grammatical_person', 'head_dependency_relation', 'head_syntactic_head_ID', 'head_word', 'number', 'out_to_in_nested_level', 'paragraph_ID', 'prop', 'sentence_ID', 'start_token', 'text']
    # Merge mention information into mention_pairs_df for each polarity
    for mention_polarity in ["A", "B"]:
        mention_pairs_df = mention_pairs_df.merge(
            CAT_entities_df[columns].add_prefix(f"{mention_polarity}_"),
            left_on=mention_polarity,
            right_index=True,
            how='left')

    default_columns = mention_pairs_df.columns

    columns = [element for element in ['mention_len', 'start_token_ID_within_sentence'] if element in features]
    # Merge mention information into mention_pairs_df for each polarity
    for mention_polarity in ["A", "B"]:
        mention_pairs_df = mention_pairs_df.merge(
            CAT_entities_df[columns].add_prefix(f"{mention_polarity}_"),
            left_on=mention_polarity,
            right_index=True,
            how='left')

    # print("get_mention_pairs_distance_features")
    mention_pairs_df = get_mention_pairs_distance_features(mention_pairs_df, features=features)
    # print("get_text_and_syntactic_match_features")
    mention_pairs_df = get_text_and_syntactic_match_features(mention_pairs_df, features=features)
    # print("get_one_hot_encoded_features")
    mention_pairs_df = get_one_hot_encoded_features(mention_pairs_df, features=features)
    # print("convert_mention_pairs_df_to_features_array")
    # Convert the enriched DataFrame into a features array
    features_array = convert_mention_pairs_df_to_features_array(mention_pairs_df, default_columns)
    return features_array


def get_coreference_resolution_training_dict(files_directory,
                                             coref_trained_model_directory,
                                             model_name="almanach/camembert-large",
                                             embedding_batch_size=10,
                                             entity_types=None,
                                             subword_pooling_strategy="average",
                                             pronoun_antecedent_max_distance=30,
                                             proper_common_nouns_antecedent_max_distance=300,
                                             features=None):
    coreference_resolution_training_dict_path = os.path.join(coref_trained_model_directory,
                                                             "coreference_resolution_training_dict.pkl")

    if not os.path.exists(coreference_resolution_training_dict_path):
        tokens_embeddings_tensor_dict_path = os.path.join(coref_trained_model_directory, "tokens_embeddings_tensor.pkl")
        tokens_embeddings_tensor_dict = get_tokens_embeddings_tensor_dict(tokens_embeddings_tensor_dict_path,
                                                                          model_name, files_directory,
                                                                          subword_pooling_strategy=subword_pooling_strategy,
                                                                          embedding_batch_size=embedding_batch_size)

        coreference_resolution_training_dict = {}
        for file_name in tqdm(tokens_embeddings_tensor_dict.keys(), desc="Generating coreference training dictionnary"):
            entities_df = load_entities_df(file_name, files_directory=files_directory, extension=".entities")
            CAT_entities_df = entities_df.copy()
            if entity_types:
                CAT_entities_df = entities_df[entities_df["cat"].isin(entity_types)].copy().reset_index(drop=True)

            tokens_embeddings_tensor = tokens_embeddings_tensor_dict[file_name]
            mentions_embeddings_tensor = get_mentions_embeddings(CAT_entities_df, tokens_embeddings_tensor)

            mention_pairs_df = initialize_mention_pairs_df(CAT_entities_df,
                                                           pronoun_antecedent_max_distance=pronoun_antecedent_max_distance,
                                                           proper_common_nouns_antecedent_max_distance=proper_common_nouns_antecedent_max_distance)
            features_array = generate_mention_pairs_features_array(mention_pairs_df,
                                                                   CAT_entities_df,
                                                                   features=features)
            labels_array = get_mention_pairs_gold_labels(mention_pairs_df,
                                                         CAT_entities_df)

            coreference_resolution_training_dict[file_name] = {'CAT_entities_df': CAT_entities_df,
                                                               'mention_pairs_df': mention_pairs_df[["A", "B"]],
                                                               'mentions_embeddings_tensor': mentions_embeddings_tensor.numpy(),
                                                               'mention_pairs_features_tensor': features_array,
                                                               'mention_pairs_labels_tensor': labels_array}

        with open(coreference_resolution_training_dict_path, 'wb') as f:
            pickle.dump(coreference_resolution_training_dict, f)

    else:
        with open(coreference_resolution_training_dict_path, 'rb') as f:
            coreference_resolution_training_dict = pickle.load(f)

    return coreference_resolution_training_dict


# All functions to prepare data for model training
def generate_split_data(files_list, model_training_dict):
    """
    Generate combined data splits for training and validation from multiple files.

    Args:
        files_list (list): List of file names to process.
        model_training_dict (dict): Dictionary containing data structures for each file,
                                    including mentions embeddings, features, labels, and mention pairs.

    Returns:
        dict: A dictionary containing the following:
            - overall_mention_pairs_df (np.ndarray): Combined mention pairs (A, B) indices.
            - overall_mentions_embeddings_tensor (np.ndarray): Combined embeddings tensor for all mentions.
            - overall_features_tensor (np.ndarray): Combined feature tensors for all mention pairs.
            - overall_labels_tensor (np.ndarray): Combined gold labels for all mention pairs.
    """
    # Initialize containers for all components
    overall_mention_pairs_dfs = []
    overall_mentions_embeddings_tensors = []
    overall_features_tensors = []
    overall_labels_tensors = []

    # Track the cumulative index offset for mentions
    overall_mention_index = 0

    for file_name in files_list:
        # Extract data from the training dictionary
        CAT_entities_df = model_training_dict[file_name]["CAT_entities_df"]
        mentions_embeddings_tensor = model_training_dict[file_name]["mentions_embeddings_tensor"]
        features_tensor = model_training_dict[file_name]["mention_pairs_features_tensor"]
        labels_tensor = model_training_dict[file_name]['mention_pairs_labels_tensor']

        # Adjust mention pair indices to account for the offset
        mention_pairs_df = model_training_dict[file_name]["mention_pairs_df"][["A", "B"]]
        mention_pairs_df = mention_pairs_df + overall_mention_index

        # Append data to the overall lists
        overall_mention_pairs_dfs.append(mention_pairs_df)
        overall_mentions_embeddings_tensors.append(mentions_embeddings_tensor)
        overall_features_tensors.append(features_tensor)
        overall_labels_tensors.append(labels_tensor)

        # Update the cumulative index offset
        overall_mention_index += len(CAT_entities_df)

        # Clear references to free up memory
        del CAT_entities_df, mentions_embeddings_tensor, features_tensor, labels_tensor, mention_pairs_df

    # Combine all data
    overall_mention_pairs_df = pd.concat(overall_mention_pairs_dfs).reset_index(drop=True)  # Convert to NumPy
    overall_mentions_embeddings_tensor = np.concatenate(overall_mentions_embeddings_tensors, axis=0)
    overall_features_tensor = np.concatenate(overall_features_tensors, axis=0)
    overall_labels_tensor = np.concatenate(overall_labels_tensors, axis=0)

    # Cleanup to reduce memory usage
    del model_training_dict, overall_mention_pairs_dfs, overall_mentions_embeddings_tensors, overall_features_tensors, overall_labels_tensors
    gc.collect()

    return {
        "overall_mention_pairs_df": overall_mention_pairs_df.to_numpy(),
        "overall_mentions_embeddings_tensor": overall_mentions_embeddings_tensor,  # NumPy array
        "overall_features_tensor": overall_features_tensor,  # NumPy array
        "overall_labels_tensor": overall_labels_tensor  # NumPy array
    }


class MentionPairsDataset(Dataset):
    def __init__(self, generator_model_data):
        """
        Initialize the dataset with generator model data.

        Args:
            generator_model_data (dict): Contains the following keys:
                - 'overall_mention_pairs_df': Mention pair indices (NumPy array).
                - 'overall_mentions_embeddings_tensor': Mention embeddings (NumPy array).
                - 'overall_features_tensor': Feature tensors (NumPy array).
                - 'overall_labels_tensor': Labels for mention pairs (NumPy array).
        """
        self.mention_pairs = generator_model_data['overall_mention_pairs_df']
        # self.per_mentions_embeddings = generator_model_data['overall_mentions_embeddings_tensor'].clone().detach().to(torch.float32)
        self.per_mentions_embeddings = torch.tensor(generator_model_data['overall_mentions_embeddings_tensor'],
                                                    dtype=torch.float32).clone()
        self.features = torch.tensor(generator_model_data['overall_features_tensor'], dtype=torch.float32).clone()
        self.labels = torch.tensor(generator_model_data['overall_labels_tensor'], dtype=torch.float32).clone()

    def __len__(self):
        """
        Returns the total number of mention pairs.
        """
        return len(self.mention_pairs)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (X, label) given an index.

        Args:
            idx (int): Index of the mention pair.

        Returns:
            tuple: (X, label)
                - X (torch.Tensor): Concatenated embeddings and features for the mention pair.
                - label (torch.Tensor): Binary label indicating coreference (0 or 1).
        """
        # Extract indices for mention pair (A, B)
        A_id, B_id = self.mention_pairs[idx]

        # Retrieve embeddings for mentions A and B
        A_embedding = self.per_mentions_embeddings[A_id]
        B_embedding = self.per_mentions_embeddings[B_id]

        # Retrieve features and label for the mention pair
        feature = self.features[idx]
        label = self.labels[idx]

        # Concatenate embeddings and features
        X = torch.cat([A_embedding, B_embedding, feature], dim=0)
        return X, label

        # # Calculate the element-wise multiplication of A and B
        # element_wise_product = A_embedding * B_embedding  # Element-wise multiplication
        # # Concatenate embeddings, dot product, and features
        # X = torch.cat([A_embedding, B_embedding, element_wise_product, feature], dim=0)
        # return X, label


def generate_train_and_validation_datasets(split, model_training_dict, train_with_validation_ratio=0.9,
                                           random_state=42):
    """
    Generates training and validation datasets by splitting the data based on the provided ratio.

    Args:
        split (dict): A dictionary containing the lists of train and validation file names.
        model_training_dict (dict): A dictionary containing data for training, including embeddings and features.
        train_with_validation_ratio (float): Ratio of training data to the total dataset (validation ratio is the complement).
        random_state (int): Random seed for reproducibility.

    Returns:
        (train_dataset, validation_dataset): Tuple of training and validation datasets.
    """
    if train_with_validation_ratio != 0:
        # Generate data for both training and validation files
        all_data = generate_split_data(split['train_files'] + split['validation_files'], model_training_dict)
        all_ids = list(range(len(all_data['overall_mention_pairs_df'])))

        # Set random seed for reproducibility
        random.seed(random_state)
        # Shuffle the list of all IDs with seed to ensure reproducibility
        random.shuffle(all_ids)

        # Split all data into train and validation based on train_with_validation_ratio
        split_index = int(train_with_validation_ratio * len(all_ids))
        train_ids = all_ids[:split_index]
        validation_ids = all_ids[split_index:]

        # Split data into train and validation sets
        train_data = all_data.copy()
        validation_data = all_data.copy()
        for key in ['overall_mention_pairs_df', 'overall_features_tensor', 'overall_labels_tensor']:
            train_data[key] = all_data[key][train_ids]
            validation_data[key] = all_data[key][validation_ids]

    else:  # If no split ratio is provided, directly use the given files for training and validation
        train_data = generate_split_data(split['train_files'], model_training_dict)
        validation_data = generate_split_data(split['validation_files'], model_training_dict)

    # Create datasets
    train_dataset = MentionPairsDataset(train_data)
    validation_dataset = MentionPairsDataset(validation_data)

    return train_dataset, validation_dataset


# Setting random state for reproduction
def set_seed(seed=42):
    """
    Sets the seed for random number generators in PyTorch, NumPy, and Python's random module.
    Enforces deterministic behavior in CuDNN for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    Seeds workers for reproducibility when using PyTorch's DataLoader with multiple workers.
    """
    np.random.seed(worker_id)
    random.seed(worker_id)

# define model class
class CoreferenceResolutionModel(nn.Module):
    """
    A feed-forward neural network for coreference resolution.

    Attributes:
        input_size (int): The size of the input feature vector.
        layers_units (int): The number of units in each hidden layer.
        layers_number (int): The number of hidden layers in the model.
        dropout (float): The dropout rate applied after each activation layer.
        layer_type (str): The type of activation function to use in hidden layers
                          (e.g., "relu", "leaky_relu", or "elu").
        network (torch.nn.Sequential): The sequential container holding the network layers.
    """

    def __init__(self, input_size, layers_units, layers_number, dropout, layer_type="relu"):
        """
        Initializes the CoreferenceResolutionModel.

        Args:
            input_size (int): Size of the input feature vector.
            layers_units (int): Number of units in each hidden layer.
            layers_number (int): Number of hidden layers.
            dropout (float): Dropout rate (value between 0 and 1).
            layer_type (str): Type of activation function. Options are "relu", "leaky_relu", or "elu".
        """
        super(CoreferenceResolutionModel, self).__init__()

        # Validate inputs
        assert layers_number > 0, "The model must have at least one hidden layer."
        assert 0 <= dropout < 1, "Dropout must be between 0 and 1."
        assert layer_type in ["relu", "leaky_relu", "elu"], "Invalid layer type. Choose 'relu', 'leaky_relu', or 'elu'."

        # Build the network layer by layer
        layers = []
        # Iterate over the desired number of hidden layers
        for i in range(layers_number):
            # Add the initial linear layer
            layers.append(nn.Linear(input_size if i == 0 else layers_units, layers_units))

            # Add the chosen activation function
            if layer_type == "relu":
                layers.append(nn.ReLU())
            elif layer_type == "leaky_relu":
                layers.append(nn.LeakyReLU(negative_slope=0.01))  # Leaky ReLU
            elif layer_type == "elu":
                layers.append(nn.ELU(alpha=1.0))  # ELU

            # Add dropout to prevent overfitting
            layers.append(nn.Dropout(dropout))

        # Add the output layer (single neuron for binary classification)
        layers.append(nn.Linear(layers_units, 1))

        # Wrap layers in a Sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1), with logits for binary classification.
        """
        return self.network(x)

def focal_loss(logits, targets, alpha=0.25, gamma=1, reduction='mean'):
    """
    Compute the Focal Loss for binary classification tasks.

    Focal loss helps to address the imbalance between well-classified
    and hard-to-classify examples by reducing the relative loss for
    well-classified examples.

    Args:
        logits (torch.Tensor): Model predictions before applying sigmoid, shape (batch_size,).
        targets (torch.Tensor): Ground truth labels, shape (batch_size,).
                                Values should be either 0 or 1.
        alpha (float, optional): Balancing factor for hard vs. easy examples (default=0.25).
        gamma (float, optional): Focusing parameter to modulate loss for easy examples (default=1.2).
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'mean' | 'sum' | 'none' (default='mean').

    Returns:
        torch.Tensor: The computed focal loss. If reduction='none', returns a tensor
                      with the same shape as `logits`. Otherwise, returns a scalar.
    """
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(logits)

    # Compute binary cross-entropy loss without reduction
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # Compute pt: the probability of the true class
    pt = torch.where(targets == 1, probs, 1 - probs)

    # Compute the modulating factor (focal weight)
    focal_weight = alpha * (1 - pt) ** gamma

    # Apply the modulating factor to the cross-entropy loss
    focal_loss = focal_weight * bce_loss

    # Apply reduction
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

def train_model(train_dataset,
                validation_dataset,
                batch_size=8000,
                layer_type="relu",
                layers_number=3,
                layers_units=1900,
                dropout=0.5,
                l2_regularization=0,
                learning_rate=0.0005,
                patience=10,
                max_epochs=100,
                verbose=True,
                focal_loss_gamma=1.2,
                focal_loss_alpha=0.5,
                random_state=None,
                loader_workers=None):
    """
    Train a coreference resolution model using focal loss and early stopping.

    Args:
        train_dataset: Training dataset.
        validation_dataset: Validation dataset.
        batch_size (int): Batch size for DataLoader.
        layers_number (int): Number of layers in the model.
        layers_units (int): Number of units per layer.
        dropout (float): Dropout rate.
        l2_regularization (float): L2 weight decay for optimizer.
        learning_rate (float): Initial learning rate for optimizer.
        patience (int): Early stopping patience.
        max_epochs (int): Maximum number of training epochs.
        verbose (int): Verbosity level.
        focal_loss_gamma (float): Gamma parameter for focal loss.
        focal_loss_alpha (float): Alpha parameter for focal loss.
        random_state (int): Random seed for reproducibility.
        loader_workers (int): Number of workers (CPU cores used) for DataLoader.

    Returns:
        model: Trained model.
        logs (pd.DataFrame): Training and validation loss/accuracy logs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    if not random_state:
        random_state = random.randint(1, 10000)
    set_seed(random_state)
    generator = torch.Generator()
    generator.manual_seed(random_state)

    # Default to using all available CPU cores for DataLoader workers
    if not loader_workers:
        loader_workers = os.cpu_count()

    # Create DataLoaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers,
                              pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers,
                            pin_memory=True, persistent_workers=True)

    # Initialize the model
    input_size = train_dataset[0][0].shape[0]  # Determine input size from one example
    model = CoreferenceResolutionModel(input_size, layers_units, layers_number, dropout, layer_type=layer_type)
    model = model.to(device)  # Move model to GPU/CPU based on availability

    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # Early stopping setup
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Initialize AMP (Automatic Mixed Precision) scaler for mixed precision training
    scaler = GradScaler()
    logs = []  # To store logs for each epoch

    # TRAINING LOOP
    for epoch in tqdm(range(max_epochs)):
        print(f"EPOCH {epoch + 1}/{max_epochs}\t\t", end="")
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Forward pass with AMP
            with autocast(device_type='cuda'):  # New syntax for autocast
                y_pred = model(X_batch).squeeze()
                loss = focal_loss(y_pred, y_batch, alpha=focal_loss_alpha, gamma=focal_loss_gamma)

            # Backward pass with scaling
            scaler.scale(loss).backward()  # Scale the loss before backward
            scaler.step(optimizer)
            scaler.update()  # Update the scaler

            # Update training loss and accuracy
            y_pred_binary = (torch.sigmoid(y_pred) > 0.5).long()
            correct_train += (y_pred_binary == y_batch).sum().item()
            total_train += y_batch.size(0)
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        # VALIDATION LOSS
        model.eval()

        # Initialize validation metrics
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad(), autocast(device_type='cuda'):
            for X_batch, y_batch in val_loader:
                # Move validation data and labels to the same device as the model
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                y_pred = model(X_batch).squeeze()
                loss = focal_loss(y_pred, y_batch, alpha=focal_loss_alpha, gamma=focal_loss_gamma)
                val_loss += loss.item() * X_batch.size(0)

                # Binary accuracy calculation
                y_pred_binary = (torch.sigmoid(y_pred) > 0.5).long()
                correct_val += (y_pred_binary == y_batch).sum().item()
                total_val += y_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_val / total_val

        # Log results
        logs.append({"epoch": epoch + 1,
                     "train_loss": train_loss,
                     "validation_loss": val_loss,
                     "train_accuracy": train_accuracy,
                     "validation_accuracy": val_accuracy})

        # Reduce learning rate and handle early stopping
        scheduler.step(val_loss)
        if best_val_loss - val_loss >= 0.0005:  # Improvement threshold
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Verbose logging
        if verbose != 0:
            print(
                f"Train Loss: {train_loss:.4f}\t\tTrain Acc:{train_accuracy:.4f}\t\tVal Loss: {val_loss:.4f}\t\tVal Acc: {val_accuracy:.4f}\t\tVal Loss Patience: {patience - patience_counter}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Convert logs to a DataFrame for easier analysis
    logs = pd.DataFrame(logs)

    return model, logs

# Model predicitons
def get_predictions(model, model_data_dict, batch_size=10000, verbose=1):
    """
    Generate predictions for coreference pairs using a trained model.

    Parameters:
        model (torch.nn.Module): The trained coreference resolution model.
        model_data_dict (dict): Dictionary containing the input features for the model.
        batch_size (int, optional): Number of examples per batch for prediction. Default is 10,000.

    Returns:
        np.ndarray: Array of predicted probabilities for each mention pair.
    """
    # Determine the appropriate device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Ensure the model is on the same device

    # Initialize dataset and data loader
    dataset = MentionPairsDataset(model_data_dict)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1,
                             pin_memory=torch.cuda.is_available()  # Optimize data transfer for GPU
                             )

    predictions = []  # List to accumulate predictions

    # Set the model to evaluation mode to disable dropout and other training-specific behavior
    model.eval()
    with torch.no_grad():  # Disable gradient computation for efficiency
        for X_batch, _ in tqdm(data_loader, desc="Predicting Coreference Pairs", leave=False, disable=(verbose != 2)):
            X_batch = X_batch.to(device)  # Move the batch to the same device as the model
            # Generate predictions and move them back to CPU for further processing
            batch_predictions = torch.sigmoid(model(X_batch)).cpu().numpy()
            predictions.append(batch_predictions)

    # Cleanup unused resources to free memory
    del data_loader, dataset
    gc.collect()

    # Concatenate all batch predictions into a single array
    return np.concatenate(predictions, axis=0)

# Posprocess mention pairs
def sort_mention_pairs_by_highest_ranked_antecedent(predicted_mention_pairs, confidence_threshold=0.2):
    """
    Sort mention pairs by the highest-ranked antecedent for each target mention.

    Parameters:
        predicted_mention_pairs (pd.DataFrame): DataFrame of predicted mention pairs with columns
            ['A', 'B', 'confidence', 'coreference_prediction'].
        confidence_threshold (float): Minimum confidence level to consider a pair for sorting.

    Returns:
        pd.DataFrame: Updated mention pairs, ensuring the highest-ranked antecedent for each target mention.
    """
    # Filter pairs based on confidence threshold and coreference prediction
    filtered_df = predicted_mention_pairs[
        (predicted_mention_pairs['confidence'] >= confidence_threshold) &
        (predicted_mention_pairs['coreference_prediction'] == 1)
        ]

    # Identify the highest-ranked antecedent (by confidence) for each target mention ('B')
    highest_ranked = filtered_df.sort_values('confidence', ascending=False).drop_duplicates(subset=['B'])

    # Merge the highest-ranked pairs with the original predicted pairs
    predicted_mention_pairs = pd.concat([highest_ranked, predicted_mention_pairs])

    # Remove duplicate mention pairs based on ['A', 'B']
    predicted_mention_pairs = predicted_mention_pairs.drop_duplicates(subset=['A', 'B'])

    return predicted_mention_pairs


def initialize_mention_pairs(entities_df):
    """
    Initialize mention pairs by assigning each mention as its own antecedent.

    Parameters:
        entities_df (pd.DataFrame): DataFrame containing entity mentions.

    Returns:
        pd.DataFrame: DataFrame with initialized mention pairs, where each mention is its own antecedent.
    """
    # Create a DataFrame where each mention is paired with itself
    initialized_pairs = pd.DataFrame({
        "A": range(len(entities_df)),
        "B": range(len(entities_df)),
        "coreference_prediction": 1
    })
    return initialized_pairs

def get_mention_pairs_gold_labels(all_processed_mention_pairs, CAT_entities_df):
    A_coref = CAT_entities_df.loc[all_processed_mention_pairs["A"].tolist()]["COREF_name"].tolist()
    B_coref = CAT_entities_df.loc[all_processed_mention_pairs["B"].tolist()]["COREF_name"].tolist()
    gold_labels = [1 if A_coref[i] == B_coref[i] else 0 for i in range(len(A_coref))]

    return gold_labels


def old_postprocess_mentions_pairs(predicted_mention_pairs,
                               CAT_entities_df=None,
                               mention_pairs_post_process=None):
    """
    Postprocess predicted mention pairs by applying a series of rules and merging with additional generated pairs.

    Parameters:
        predicted_mention_pairs (pd.DataFrame): DataFrame containing predicted mention pairs with confidence scores.
        CAT_entities_df (pd.DataFrame, optional): DataFrame of categorized entities for rule-based processing.
        mention_pairs_post_process (list of str, optional): List of postprocessing steps to apply.
            Options include:
            - 'sort_by_confidence': Sort mention pairs by confidence scores.
            - 'sort_by_highest_ranked_antecedent': Sort pairs by the rank of the highest-ranked antecedent.
            - 'assign_coreference_to_proper_mentions': Assign coreference to proper mentions nested within other mentions.
            - 'assign_coreference_to_direct_nested_mentions': Assign coreference to directly nested mentions.
            - 'assign_coreference_to_explicit_mentions': Assign coreference to explicitly defined mentions.

    Returns:
        pd.DataFrame: Processed mention pairs with added labels and error metrics.
    """
    # Default postprocessing steps if none are provided
    if mention_pairs_post_process is None:
        mention_pairs_post_process = [
            'sort_by_confidence',
            'sort_by_highest_ranked_antecedent',
            'assign_coreference_to_proper_mentions',
            'assign_coreference_to_direct_nested_mentions',
            'assign_coreference_to_explicit_mentions'
        ]
    # Sort predicted pairs by confidence if specified
    if 'sort_by_confidence' in mention_pairs_post_process:
        predicted_mention_pairs = predicted_mention_pairs.sort_values('confidence', ascending=False).reset_index(
            drop=True)

    # Sort predicted pairs by highest confidence antecedent for each mention
    if 'sort_by_highest_ranked_antecedent' in mention_pairs_post_process:
        predicted_mention_pairs = sort_mention_pairs_by_highest_ranked_antecedent(predicted_mention_pairs,
                                                                                  confidence_threshold=0.0)

    # Initialize list for rule-based mention pairs
    post_processed_mention_pairs = [initialize_mention_pairs(CAT_entities_df)]

    # Apply additional rules based on the provided configuration
    if 'assign_coreference_to_proper_mentions' in mention_pairs_post_process:
        post_processed_mention_pairs.append(assign_coreference_to_proper_mentions_if_A_is_part_of_B(CAT_entities_df))
    if 'assign_coreference_to_direct_nested_mentions' in mention_pairs_post_process:
        post_processed_mention_pairs.append(assign_coreference_to_direct_nested_mentions(CAT_entities_df))
    if 'assign_coreference_to_explicit_mentions' in mention_pairs_post_process:
        post_processed_mention_pairs.append(
            assign_coreference_to_explicit_proper_and_noun_phrase_mentions(CAT_entities_df))

    # Combine rule-based mention pairs into a single DataFrame
    post_processed_mention_pairs = pd.concat(post_processed_mention_pairs)
    post_processed_mention_pairs['confidence'] = 1  # Assign full confidence to rule-based pairs

    # Combine rule-based and predicted mention pairs, dropping duplicates
    all_processed_mention_pairs = pd.concat([post_processed_mention_pairs, predicted_mention_pairs]).reset_index(
        drop=True)

    # Add gold labels and compute errors
    all_processed_mention_pairs = all_processed_mention_pairs.drop_duplicates(subset=['A', 'B'])
    if "COREF" in CAT_entities_df.columns:
        all_processed_mention_pairs['labels'] = get_mention_pairs_gold_labels(all_processed_mention_pairs,
                                                                              CAT_entities_df)
        all_processed_mention_pairs['error'] = abs(
            all_processed_mention_pairs['coreference_prediction'] - all_processed_mention_pairs['labels'])

    return all_processed_mention_pairs

def initialize_gold_coreference_matrix_from_entities_df(entities_df, coref_column="COREF"):
    """
    Initialize a gold coreference matrix from the 'COREF' column of the entities DataFrame.
    """
    # Convert to NumPy array (avoiding an extra copy)
    COREF_array = entities_df[coref_column].to_numpy()

    # Use np.equal.outer() for string-based comparison
    gold_coreference_matrix = np.equal.outer(COREF_array, COREF_array)

    # Convert boolean values to int8 (-1, 1) for memory efficiency
    return np.where(gold_coreference_matrix, np.int8(1), np.int8(-1))

def generate_coreference_matrix_with_cache(entities_df, mention_pairs_df, verbose=1, confidence_threshold=0):
    """
    Generates a predicted coreference matrix by iteratively clustering mention pairs
    and propagating coreference relations to clusters.

    Parameters:
        entities_df (pd.DataFrame): DataFrame containing mentions with entity IDs.
        mention_pairs_df (pd.DataFrame): DataFrame containing mention pairs and their coreference predictions.
        verbose (int, optional): If non-zero, enables verbose output for debugging.

    Returns:
        np.ndarray: A coreference matrix where:
                    - `1` indicates that two mentions belong to the same coreference cluster.
                    - `-1` indicates that two mentions do not belong to the same coreference cluster.
    """
    N = len(entities_df)  # Number of mentions
    matrix = np.zeros((N, N), dtype=np.int8)  # Use int8 to save memory (since values are only -1, 0, 1)
    np.fill_diagonal(matrix, 1)  # Diagonal elements represent a mention referring to itself (coreference = 1)

    mention_pairs_df["confidence"].fillna(1, inplace=True)
    mention_pairs_df = mention_pairs_df[mention_pairs_df["confidence"] >= confidence_threshold].copy().reset_index(drop=True)
    # Replace '0' with '-1' in the coreference predictions
    mention_pairs_df['coreference_prediction'].replace(0, -1, inplace=True)

    # Convert mention pairs to a list of tuples (A, B, prediction)
    mention_pairs_list = list(mention_pairs_df[['A', 'B', 'coreference_prediction']].itertuples(index=False, name=None))

    # Initialize a cache to quickly access coreference IDs for mentions
    coref_cache = {i: {i} for i in range(N)}

    # Process each mention pair (A, B, prediction)
    if verbose == 0:
        for i, (A, B, prediction) in enumerate(mention_pairs_list):

            # if i % 100000 == 0:
            #     gc.collect()

            if matrix[A, B] != 0:  # Skip already processed pairs
                continue

            A_coref_ids = coref_cache[A]
            B_coref_ids = coref_cache[B]

            # Merge the coreference IDs for A and B
            AB_concat_ids = A_coref_ids | B_coref_ids  # Set union to combine coreference IDs
            AB_concat_ids_list = list(AB_concat_ids)  # Convert once
            A_coref_ids_list = list(A_coref_ids)
            B_coref_ids_list = list(B_coref_ids)
            AB_matrix = matrix[np.ix_(A_coref_ids_list, B_coref_ids_list)]

            empty_matrix = not AB_matrix.any()  # Check if the submatrix is empty
            coreference_pairs = (AB_matrix == 1).any()  # Check if any pairs in the submatrix are coreferent
            non_coreference_pairs = (AB_matrix == -1).any()  # Check if any pairs in the submatrix are not coreferent

            # Handle different scenarios based on the current matrix and prediction
            if empty_matrix:
                if prediction == 1:
                    matrix[np.ix_(AB_concat_ids_list, AB_concat_ids_list)] = 1  # Propagate coreference
                    for idx in AB_concat_ids:
                        coref_cache[idx] = AB_concat_ids  # Update cache with the new coreference group
                else:  # prediction == -1
                    matrix[np.ix_(A_coref_ids_list, B_coref_ids_list)] = -1  # Propagate non-coreference
                    matrix[np.ix_(B_coref_ids_list, A_coref_ids_list)] = -1
            elif non_coreference_pairs and not coreference_pairs:
                if prediction == -1:
                    matrix[np.ix_(A_coref_ids_list, B_coref_ids_list)] = -1  # No coreference
                    matrix[np.ix_(B_coref_ids_list, A_coref_ids_list)] = -1
                elif verbose:
                    # gc.collect()
                    print(f'{i}\tIllegal Coreference {[A, B]}')  # Log error if prediction contradicts existing coreference
            elif coreference_pairs and not non_coreference_pairs:
                if prediction == 1:
                    matrix[np.ix_(AB_concat_ids_list, AB_concat_ids_list)] = 1  # Merge coreference clusters
                    for idx in AB_concat_ids:
                        coref_cache[idx] = AB_concat_ids  # Update cache
                elif verbose:
                    print(f'{i}\tIllegal Breaking Coreference {[A, B]}')  # Log error if prediction breaks existing coreference
            elif non_coreference_pairs and coreference_pairs:
                print("Clustering ERROR")  # This scenario should not happen
                break

    elif verbose == 1:
        for i, (A, B, prediction) in enumerate(tqdm(mention_pairs_list, desc="Generating coreference matrix", leave=False)):

            # if i % 100000 == 0:
            #     gc.collect()

            if matrix[A, B] != 0:  # Skip already processed pairs
                continue

            A_coref_ids = coref_cache[A]
            B_coref_ids = coref_cache[B]

            # Merge the coreference IDs for A and B
            AB_concat_ids = A_coref_ids | B_coref_ids  # Set union to combine coreference IDs
            AB_concat_ids_list = list(AB_concat_ids)  # Convert once
            A_coref_ids_list = list(A_coref_ids)
            B_coref_ids_list = list(B_coref_ids)
            AB_matrix = matrix[np.ix_(A_coref_ids_list, B_coref_ids_list)]

            empty_matrix = not AB_matrix.any()  # Check if the submatrix is empty
            coreference_pairs = (AB_matrix == 1).any()  # Check if any pairs in the submatrix are coreferent
            non_coreference_pairs = (
                        AB_matrix == -1).any()  # Check if any pairs in the submatrix are not coreferent

            # Handle different scenarios based on the current matrix and prediction
            if empty_matrix:
                if prediction == 1:
                    matrix[np.ix_(AB_concat_ids_list, AB_concat_ids_list)] = 1  # Propagate coreference
                    for idx in AB_concat_ids:
                        coref_cache[idx] = AB_concat_ids  # Update cache with the new coreference group
                else:  # prediction == -1
                    matrix[np.ix_(A_coref_ids_list, B_coref_ids_list)] = -1  # Propagate non-coreference
                    matrix[np.ix_(B_coref_ids_list, A_coref_ids_list)] = -1
            elif non_coreference_pairs and not coreference_pairs:
                if prediction == -1:
                    matrix[np.ix_(A_coref_ids_list, B_coref_ids_list)] = -1  # No coreference
                    matrix[np.ix_(B_coref_ids_list, A_coref_ids_list)] = -1
                elif verbose:
                    # gc.collect()
                    print(
                        f'{i}\tIllegal Coreference {[A, B]}')  # Log error if prediction contradicts existing coreference
            elif coreference_pairs and not non_coreference_pairs:
                if prediction == 1:
                    matrix[np.ix_(AB_concat_ids_list, AB_concat_ids_list)] = 1  # Merge coreference clusters
                    for idx in AB_concat_ids:
                        coref_cache[idx] = AB_concat_ids  # Update cache
                elif verbose:
                    print(
                        f'{i}\tIllegal Breaking Coreference {[A, B]}')  # Log error if prediction breaks existing coreference
            elif non_coreference_pairs and coreference_pairs:
                print("Clustering ERROR")  # This scenario should not happen
                break

    return matrix

def extract_mentions_and_links_from_coreference_matrix(matrix):
    """
    Extracts mentions and links from the given coreference matrix.

    Args:
    - matrix (np.ndarray): The coreference matrix (2D numpy array), where 1 represents coreference and -1 represents non-coreference.

    Returns:
    - mentions (list): A list of mentions (indices) in the matrix.
    - minimal_links (list): A list of pairs of mentions that are linked based on coreference.
    """
    matrix_size = matrix.shape[0]
    mentions = list(range(matrix_size))  # Each index in the matrix is a mention
    minimal_links = []  # List to store mention pairs that are coreferent
    treated_mentions = set()  # To track mentions that have been processed

    for i in range(matrix_size):
        minimal_links.append([i, i])  # Each mention is coreferent with itself

        if i not in treated_mentions:  # Process only mentions that haven't been processed yet
            treated_mentions.add(i)

            # Check for coreferences between mention i and other mentions
            for j in range(i + 1, matrix_size):
                if matrix[i, j] == 1:  # If mentions i and j are coreferent
                    minimal_links.append([i, j])  # Add the pair to the coreference links
                    treated_mentions.add(j)  # Mark mention j as processed

    return mentions, minimal_links

def coreference_resolution_metrics(gold_coreference_matrix, predicted_coreference_matrix):
    """
    Compute coreference resolution metrics (MUC, B3, CEAF) by comparing gold and predicted coreference matrices.
    This function extracts mentions and links, computes metrics using Scorch, and returns a DataFrame with precision, recall, and F1 scores.

    Parameters:
        gold_coreference_matrix (np.ndarray): Gold standard coreference matrix.
        predicted_coreference_matrix (np.ndarray): Predicted coreference matrix.

    Returns:
        pd.DataFrame: A DataFrame containing recall, precision, and F1 scores for coreference metrics.
    """
    # Initialize dictionaries for gold and predicted data
    gold = {"type": "graph", "mentions": [], "links": []}
    predicted = {"type": "graph", "mentions": [], "links": []}

    # Extract mentions and links from the gold and predicted coreference matrices
    gold['mentions'], gold['links'] = extract_mentions_and_links_from_coreference_matrix(gold_coreference_matrix)
    predicted['mentions'], predicted['links'] = extract_mentions_and_links_from_coreference_matrix(
        predicted_coreference_matrix)

    # Save the gold and predicted dictionaries as JSON files for Scorch
    with open('gold.json', 'w') as gold_file:
        json.dump(gold, gold_file, indent=4)
    with open('predicted.json', 'w') as predicted_file:
        json.dump(predicted, predicted_file, indent=4)

    # Run Scorch to compute coreference metrics
    command = ["scorch", "gold.json", "predicted.json"]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

    # Process the output from Scorch
    input_string = result.stdout
    lines = input_string.strip().split('\n')

    # Initialize an empty dictionary to store the metrics
    result_dict = {}

    # Iterate over each line to extract metrics
    for line in lines:
        if ':' in line:
            key, metrics = line.split(':', 1)  # Split the key and metrics part
            metrics_dict = {}

            # Split the metrics into individual components (recall, precision, F1 score)
            for metric in metrics.strip().split('\t'):
                if '=' in metric:  # Only process valid key=value pairs
                    metric_key, metric_value = metric.split('=')
                    metrics_dict[metric_key.strip()] = float(metric_value)

            # Add the extracted metrics to the result dictionary
            result_dict[key.strip()] = metrics_dict

    # Convert the result dictionary to a DataFrame for better presentation
    coreference_metrics_df = pd.DataFrame(result_dict).T
    coreference_metrics_df.columns = ['recall', 'precision', 'f1_score']

    # Reorder columns and rename index for clarity
    coreference_metrics_df = coreference_metrics_df[['recall', 'precision', 'f1_score']]
    coreference_metrics_df.rename(index={"B³": "B3"}, inplace=True)

    # Add a 'CONLL' row, which is the mean of the MUC, B3, and CEAF_e scores
    coreference_metrics_df.loc['CONLL'] = coreference_metrics_df.loc[['MUC', 'B3', 'CEAF_e']].mean()

    return coreference_metrics_df

def get_coref_metrics_for_token_window(start_token_id=0,
                                       end_token_id=None,
                                       CAT_entities_df=None,
                                       all_processed_mention_pairs=None):
    """
    Calculate coreference metrics for a specific window of tokens.

    Parameters:
        start_token_id (int): The starting token ID for the window (inclusive).
        end_token_id (int): The ending token ID for the window (inclusive). If None, it defaults to the maximum token ID.
        CAT_entities_df (pd.DataFrame): DataFrame containing mention entities with token spans ('start_token' and 'end_token').
        all_processed_mention_pairs (pd.DataFrame): DataFrame containing processed mention pairs with indices 'A' and 'B'.

    Returns:
        tuple:
            - (int): Number of valid entities within the token window.
            - (pd.DataFrame): Coreference metrics for the specified token window.
    """
    # Step 1: Set default value for end_token_id if not provided
    if end_token_id == None:
        end_token_id = CAT_entities_df["end_token"].max()

    # Step 2: Filter mentions within the specified token window
    valid_CAT_entities_df = CAT_entities_df.copy()
    valid_CAT_entities_df = valid_CAT_entities_df[
        (valid_CAT_entities_df["end_token"] >= start_token_id)  # Mention ends after or at start_token_id
        & (valid_CAT_entities_df["start_token"] <= end_token_id)  # Mention starts before or at end_token_id
        ]

    # Step 3: Reset the index of filtered mentions and retain the old index for mapping
    valid_CAT_entities_df = valid_CAT_entities_df.reset_index(names="old_index")

    # Step 4: Create a mapping of old indices to new indices in the filtered DataFrame
    index_mapping = dict(zip(valid_CAT_entities_df["old_index"], valid_CAT_entities_df.index))

    # Step 5: Filter mention pairs to retain only those involving both mentions in the token window
    valid_processed_mention_pairs = all_processed_mention_pairs.copy()
    valid_processed_mention_pairs = valid_processed_mention_pairs[
        (valid_processed_mention_pairs["A"].isin(index_mapping.keys())) &
        (valid_processed_mention_pairs["B"].isin(index_mapping.keys()))
        ]

    # Step 6: Update the 'A' and 'B' columns of mention pairs to use the new indices
    valid_processed_mention_pairs["A"] = valid_processed_mention_pairs["A"].map(index_mapping)
    valid_processed_mention_pairs["B"] = valid_processed_mention_pairs["B"].map(index_mapping)

    # Step 7: Generate the gold and predicted coreference matrices
    gold_coreference_matrix = initialize_gold_coreference_matrix_from_entities_df(valid_CAT_entities_df)
    predicted_coreference_matrix = generate_coreference_matrix_with_cache(valid_CAT_entities_df,
                                                                          valid_processed_mention_pairs,
                                                                          verbose=0,
                                                                          confidence_threshold=0)

    # Step 8: Calculate coreference metrics between the gold and predicted matrices
    coreference_metrics_df = coreference_resolution_metrics(gold_coreference_matrix, predicted_coreference_matrix)

    # Return the number of valid mentions and the calculated metrics
    return len(valid_CAT_entities_df), coreference_metrics_df


def evaluate_coreference_model(trained_coreference_model, coreference_resolution_training_dict, test_file,
                               files_directory,
                               mention_pairs_post_process=['sort_by_confidence', 'sort_by_highest_ranked_antecedent',
                                                           'assign_coreference_to_proper_mentions',
                                                           'assign_coreference_to_direct_nested_mentions',
                                                           'assign_coreference_to_explicit_mentions'],
                               verbose=1,
                               ):
    """
    Evaluates a trained coreference model on a given test file and calculates coreference metrics for different token windows.

    Args:
    - trained_coreference_model: The trained model to evaluate.
    - coreference_resolution_training_dict: A dictionary containing training data.
    - test_file: The test file to evaluate on.
    - files_directory: The directory where files are stored.
    - mention_pairs_post_process: A list of post-processing steps to apply to predicted mention pairs.

    Returns:
    - test_prediction_report (pd.DataFrame): A dataframe containing evaluation metrics (precision, recall, F1) for the predictions.
    - model_coreference_resolution_metrics_df (pd.DataFrame): A dataframe containing coreference metrics for different token windows.
    """
    # Step 1: Load entities and token data
    CAT_entities_df = coreference_resolution_training_dict[test_file]['CAT_entities_df']
    tokens_df = load_tokens_df(test_file, files_directory)

    # Step 2: Prepare the test data and gold labels
    test_data = generate_split_data([test_file], coreference_resolution_training_dict)
    gold_labels = test_data["overall_labels_tensor"]

    # Step 3: Get model predictions
    predictions = get_predictions(trained_coreference_model, test_data, batch_size=10000, verbose=verbose)
    argmax_predictions = np.where(predictions >= 0.5, 1, 0)

    # Step 4: Generate prediction report (precision, recall, F1) for predicted mentions pairs
    test_prediction_report = pd.DataFrame(
        classification_report(np.array(gold_labels), argmax_predictions, digits=4, output_dict=True)).T

    # Step 5: Process predicted mention pairs
    predicted_mention_pairs = pd.DataFrame(test_data['overall_mention_pairs_df'], columns=['A', 'B'])
    predicted_mention_pairs['coreference_prediction'] = predictions
    predicted_mention_pairs['confidence'] = abs(predicted_mention_pairs['coreference_prediction'].copy() - 0.5) * 2
    predicted_mention_pairs['coreference_prediction'] = (predicted_mention_pairs['coreference_prediction'] >= 0.5) * 1
    predicted_mention_pairs['labels'] = gold_labels
    predicted_mention_pairs['error'] = abs(
        predicted_mention_pairs['coreference_prediction'] - predicted_mention_pairs['labels'])

    # Step 6: Postprocess mentions
    all_processed_mention_pairs = old_postprocess_mentions_pairs(predicted_mention_pairs,
                                                             CAT_entities_df=CAT_entities_df,
                                                             mention_pairs_post_process=mention_pairs_post_process)

    # Step 7: Calculate coreference metrics for different token windows
    model_coreference_resolution_metrics = []
    tokens_count = len(tokens_df)
    tokens_span_step = 10000
    window_ranges = [tokens_count]

    # Iterate over different window sizes
    for window_range in tqdm(window_ranges, desc="Evaluating Coreference Resolution for different windows",
                             leave=False, disable=(verbose != 2)):
        # Generates window boundaries based on the token count and window size
        boundaries = [{"start_token_id": i, "end_token_id": i + window_range}
                      for i in range(0, tokens_count, window_range) if i + window_range <= tokens_count
                      ]

        for window in tqdm(boundaries, desc=f"window_range: {window_range} Tokens", leave=False, disable=(verbose != 2)):
            start_token_id = window['start_token_id']
            end_token_id = window['end_token_id']
            tokens_span = end_token_id - start_token_id

            # Step 8: Get coreference metrics for each window
            mentions_count, coreference_metrics_df = get_coref_metrics_for_token_window(start_token_id=start_token_id,
                                                                                        end_token_id=end_token_id,
                                                                                        CAT_entities_df=CAT_entities_df,
                                                                                        all_processed_mention_pairs=all_processed_mention_pairs)
            # Step 9: Collect the coreference resolution metrics
            model_coreference_resolution_metrics.append({"file_name": test_file,
                                                         "tokens_span": tokens_span,
                                                         "mentions_count": mentions_count,
                                                         "start_token_id": start_token_id,
                                                         "end_token_id": end_token_id,
                                                         "MUC_recall": coreference_metrics_df.loc["MUC", "recall"],
                                                         "MUC_precision": coreference_metrics_df.loc[
                                                             "MUC", "precision"],
                                                         "MUC_f1": coreference_metrics_df.loc["MUC", "f1_score"],
                                                         "B3_recall": coreference_metrics_df.loc["B3", "recall"],
                                                         "B3_precision": coreference_metrics_df.loc["B3", "precision"],
                                                         "B3_f1": coreference_metrics_df.loc["B3", "f1_score"],
                                                         "CEAFe_recall": coreference_metrics_df.loc["CEAF_e", "recall"],
                                                         "CEAFe_precision": coreference_metrics_df.loc[
                                                             "CEAF_e", "precision"],
                                                         "CEAFe_f1": coreference_metrics_df.loc["CEAF_e", "f1_score"],
                                                         "CONLL_recall": coreference_metrics_df.loc["CONLL", "recall"],
                                                         "CONLL_precision": coreference_metrics_df.loc[
                                                             "CONLL", "precision"],
                                                         "CONLL_f1": coreference_metrics_df.loc["CONLL", "f1_score"],
                                                         })

    # Convert collected metrics to DataFrame
    model_coreference_resolution_metrics_df = pd.DataFrame(model_coreference_resolution_metrics)

    return test_prediction_report, model_coreference_resolution_metrics_df

def coreference_resolution_LOOCV_full_model_training(
        files_directory=None,
        model_name="almanach/camembert-large",
        train_files="all",
        test_splits="LOOCV",
        train_final_model = True,
        coref_trained_model_directory=None,
        pronoun_antecedent_max_distance=30,
        proper_common_nouns_antecedent_max_distance=300,
        entity_types=None,
        subword_pooling_strategy="first_last", # ["average", "first", "last", "first_last", "max"]
        features=['mention_len', 'start_token_ID_within_sentence', 'mention_ID_delta', 'start_token_delta',
                  'end_token_delta', 'paragraph_ID_delta', 'sentence_ID_delta', 'out_to_in_nested_level_delta',
                  'shared_token_ratio', 'text_match', 'head_text_match', 'syntactic_head_match', 'cat_match', 'prop',
                  'head_dependency_relation', 'gender', 'number', 'grammatical_person'],
        embedding_batch_size=10,
        batch_size=16000,
        layers_number=3,
        layers_units=1900,
        dropout=0.6,
        l2_regularization=0,
        learning_rate=0.0004,
        patience=10,
        max_epochs=75,
        verbose=2,
        focal_loss_gamma=1,
        focal_loss_alpha=0.25,
        layer_type="relu",  # relu, leaky_relu, elu
        train_with_validation_ratio=0.85,
        random_state=None,
        loader_workers=8,
        mention_pairs_post_process=['sort_by_confidence',
                                    'sort_by_highest_ranked_antecedent',
                                    'assign_coreference_to_proper_mentions',
                                    'assign_coreference_to_direct_nested_mentions',
                                    'assign_coreference_to_explicit_mentions'],
        evaluate=True,
):
    config = AutoConfig.from_pretrained(model_name)
    embedding_dim = config.hidden_size

    if coref_trained_model_directory == None:
        coref_trained_model_directory = os.path.join(files_directory,
                                                     f"coreference_resolution_model_{model_name.split('/')[-1]}_{pronoun_antecedent_max_distance}_{proper_common_nouns_antecedent_max_distance}_{'_'.join(sorted(entity_types))}")

    if not os.path.exists(coref_trained_model_directory):
        os.makedirs(coref_trained_model_directory)
    print(f"Training Directory:\n{coref_trained_model_directory}")

    coreference_resolution_training_dict = get_coreference_resolution_training_dict(
        files_directory,
        coref_trained_model_directory=coref_trained_model_directory,
        model_name=model_name,
        pronoun_antecedent_max_distance=pronoun_antecedent_max_distance,
        proper_common_nouns_antecedent_max_distance=proper_common_nouns_antecedent_max_distance,
        entity_types=entity_types,
        subword_pooling_strategy=subword_pooling_strategy,
        embedding_batch_size=embedding_batch_size,
        features=features)

    # 5. File Splits for Cross-Validation
    if train_files == "all":
        all_files = list(coreference_resolution_training_dict.keys())
    if isinstance(train_files, list):
        all_files = [file_name for file_name in train_files if file_name in train_files]

    if test_splits == "LOOCV":
        test_splits = {file_name: [file_name] for file_name in all_files}

    if isinstance(test_splits, list): # test_splits contains a list
        # Case: list of lists (e.g. predefined folds)
        if test_splits and isinstance(test_splits[0], list):
            print("test_splits is a list of lists")

            # Map index → list of files in that fold
            test_splits = {str(i): split_files for i, split_files in enumerate(test_splits)}
        else:
            test_splits = {file_name: [file_name] for file_name in test_splits}

    if not isinstance(test_splits, dict):
        print(f"test_splits is not a dictionary.")
        sys.exit(1)

    if train_final_model == True:
        test_splits["FINAL_MODEL"] = []

    to_process_splits = {}
    for split_name, split_files in test_splits.items():
        trained_coreference_model_path = os.path.join(coref_trained_model_directory, f"coreference_resolution_{split_name}")
        if not os.path.isfile(trained_coreference_model_path):
            to_process_splits[trained_coreference_model_path] = split_files

    for trained_coreference_model_path, test_files in tqdm(to_process_splits.items()):

        validation_files = []
        train_files = [file for file in all_files if file not in test_files + validation_files]

        print(f"Trained Coreference Model Path:\n{trained_coreference_model_path}")

        split = {"test_files": test_files,
                 "validation_files": validation_files,
                 "train_files": train_files}

        if validation_files != []:  # Corrected check for an empty list
            train_with_validation_ratio = 0

        train_dataset, validation_dataset = generate_train_and_validation_datasets(
            split,
            coreference_resolution_training_dict,
            train_with_validation_ratio=train_with_validation_ratio,
            random_state=random_state)

        model, logs = train_model(train_dataset=train_dataset,
                                  validation_dataset=validation_dataset,
                                  batch_size=batch_size,
                                  layer_type=layer_type,
                                  layers_number=layers_number,
                                  layers_units=layers_units,
                                  dropout=dropout,
                                  l2_regularization=l2_regularization,
                                  learning_rate=learning_rate,
                                  patience=patience,
                                  max_epochs=max_epochs,
                                  verbose=verbose,
                                  focal_loss_gamma=focal_loss_gamma,
                                  focal_loss_alpha=focal_loss_alpha,
                                  random_state=random_state,
                                  loader_workers=loader_workers)

        all_test_prediction_reports, all_model_coreference_resolution_metrics_dfs = None, None
        if len(test_files) != 0 and evaluate==True:
            all_test_prediction_reports, all_model_coreference_resolution_metrics_dfs = {}, {}
            for test_file in test_files:
                test_prediction_report, model_coreference_resolution_metrics_df = evaluate_coreference_model(
                    trained_coreference_model=model,
                    coreference_resolution_training_dict=coreference_resolution_training_dict,
                    test_file=test_file,
                    files_directory=files_directory,
                    mention_pairs_post_process=mention_pairs_post_process,
                    verbose=verbose,)
                all_test_prediction_reports[test_file] = test_prediction_report
                all_model_coreference_resolution_metrics_dfs[test_file] = model_coreference_resolution_metrics_df

        # Prepare a dictionary to store all relevant training and evaluation information
        trained_model_infos = {"files_directory": files_directory,
                               "base_model_name": model_name,
                               "entity_types": entity_types,
                               "subword_pooling_strategy": subword_pooling_strategy,
                               "all_files": all_files,
                               "test_files": test_files,
                               "train_with_validation_ratio": train_with_validation_ratio,
                               "pronoun_antecedent_max_distance": pronoun_antecedent_max_distance,
                               "proper_common_nouns_antecedent_max_distance": proper_common_nouns_antecedent_max_distance,
                               "features": features,
                               "batch_size": batch_size,
                               "embedding_dim": embedding_dim,
                               "layers_number": layers_number,
                               "layers_units": layers_units,
                               "dropout": dropout,
                               "l2_regularization": l2_regularization,
                               "learning_rate": learning_rate,
                               "patience": patience,
                               "max_epochs": max_epochs,
                               "focal_loss_gamma": focal_loss_gamma,
                               "focal_loss_alpha": focal_loss_alpha,
                               "layer_type": layer_type,
                               "random_state": random_state,
                               "model": model,
                               "logs": logs,
                               "mention_pairs_post_process": mention_pairs_post_process,
                               "all_test_prediction_reports": all_test_prediction_reports,
                               "all_model_coreference_resolution_metrics_dfs": all_model_coreference_resolution_metrics_dfs,
                               }
        # Save the trained model information and evaluation results to a pickle file
        with open(trained_coreference_model_path, "wb") as file:
            pickle.dump(trained_model_infos, file)


def generate_corpus_infos_table(trained_model_infos,
                                model_evaluation_files):
    all_files = trained_model_infos["all_files"]
    files_directory = trained_model_infos["files_directory"]

    ## Collect general information about the training corpus
    training_corpus_infos = []
    total_tokens_count = 0

    # Process each training file to count tokens and determine evaluation status
    for file_name in all_files:
        tokens_df = load_tokens_df(file_name, files_directory)
        total_tokens_count += len(tokens_df)
        is_in_model_eval = "**True**" if file_name in model_evaluation_files else "False"
        training_corpus_infos.append({"Document": file_name, "Tokens Count": f"{len(tokens_df):,} tokens",
                                      "Is included in model eval": is_in_model_eval})

    # Summarize training corpus statistics
    training_corpus_infos_df = pd.DataFrame(training_corpus_infos)
    training_corpus_infos_df.loc[len(training_corpus_infos_df), ["Document", "Tokens Count",
                                                                 "Is included in model eval"]] = "TOTAL", f"{total_tokens_count:,} tokens", f"{len(model_evaluation_files)} files used for cross-validation"
    corpus_infos_table = tabulate(training_corpus_infos_df, headers="keys", tablefmt="github")

    return corpus_infos_table

def generate_overall_LOOCV_metrics_df(
        coref_trained_model_directory,
        trained_model_infos,
        model_evaluation_files="all",
        metrics_columns=["MUC_f1", "B3_f1", "CEAFe_f1", "CONLL_f1"],
        token_windows_length=[500, 1000, 2000, 5000, 10000, 25000, 50000]):
    if model_evaluation_files == "all":
        model_evaluation_files = trained_model_infos["all_files"]

    trained_models = [file for file in os.listdir(coref_trained_model_directory) if file in model_evaluation_files]

    overall_coreference_resolution_metrics_dfs = []
    for file_name in tqdm(trained_models, desc="Evaluating Cross-Validation Models"):
        model_path = os.path.join(coref_trained_model_directory, file_name)
        with open(model_path, "rb") as file:
            trained_model_infos = pickle.load(file)

        model_coreference_resolution_metrics_df = trained_model_infos["model_coreference_resolution_metrics_df"]
        overall_coreference_resolution_metrics_dfs.append(model_coreference_resolution_metrics_df)

    overall_coreference_resolution_metrics_df = pd.concat(overall_coreference_resolution_metrics_dfs)

    # Get the coreference metrics for non-overlapping windows of different length
    all_tokens_spans_metrics = []
    for tokens_span in token_windows_length:
        filtered_metrics_df = overall_coreference_resolution_metrics_df.copy()
        filtered_metrics_df = filtered_metrics_df[["file_name", "tokens_span", "mentions_count"] + metrics_columns]
        filtered_metrics_df = filtered_metrics_df[filtered_metrics_df['tokens_span'] == tokens_span]

        documents_count = len(filtered_metrics_df['file_name'].unique())
        text_count = len(filtered_metrics_df)

        metrics_dict = {"tokens_span": tokens_span,
                        "documents_count": documents_count,
                        "text_count": text_count
                        }
        for column in metrics_columns:
            metrics_dict[column] = filtered_metrics_df[column].mean()
        all_tokens_spans_metrics.append(metrics_dict)

    non_overlapping_windows_coref_metrics_df = pd.DataFrame(all_tokens_spans_metrics)

    # Get the coreference metrics for the fully annotated sample of each document
    full_annotated_sample_coref_metrics_df = (
        overall_coreference_resolution_metrics_df
        .sort_values(["file_name", "tokens_span", "CONLL_f1"], ascending=[True, False, False])
        .drop_duplicates(subset="file_name", keep="first")
    ).reset_index(drop=True)
    # Filtering columns of interest
    full_annotated_sample_coref_metrics_df = full_annotated_sample_coref_metrics_df[
        ["file_name", "tokens_span", "mentions_count"] + metrics_columns]

    # Generate generate_corpus_infos_table
    corpus_infos_table = generate_corpus_infos_table(trained_model_infos,
                                                     model_evaluation_files)

    return trained_model_infos, overall_coreference_resolution_metrics_df, non_overlapping_windows_coref_metrics_df, full_annotated_sample_coref_metrics_df, corpus_infos_table


def generate_datacard_metrics_tables(non_overlapping_windows_coref_metrics_df, full_annotated_sample_coref_metrics_df):
    non_overlapping_windows_coref_metrics_df.columns = ['Window width (tokens)', 'Document count', 'Sample count',
                                                        'MUC F1', 'B3 F1',
                                                        'CEAFe F1', 'CONLL F1']
    non_overlapping_windows_coref_metrics_df = non_overlapping_windows_coref_metrics_df[
        non_overlapping_windows_coref_metrics_df['Document count'] > 0].copy()
    # Format the columns
    non_overlapping_windows_coref_metrics_df['Window width (tokens)'] = non_overlapping_windows_coref_metrics_df[
        'Window width (tokens)'].apply(lambda x: f"{x:,}")
    non_overlapping_windows_coref_metrics_df['Sample count'] = non_overlapping_windows_coref_metrics_df[
        'Sample count'].apply(lambda x: f"{x:,}")
    non_overlapping_windows_coref_metrics_df['MUC F1'] = non_overlapping_windows_coref_metrics_df['MUC F1'].apply(
        lambda x: f"{x:.2%}")
    non_overlapping_windows_coref_metrics_df['B3 F1'] = non_overlapping_windows_coref_metrics_df['B3 F1'].apply(
        lambda x: f"{x:.2%}")
    non_overlapping_windows_coref_metrics_df['CEAFe F1'] = non_overlapping_windows_coref_metrics_df['CEAFe F1'].apply(
        lambda x: f"{x:.2%}")
    non_overlapping_windows_coref_metrics_df['CONLL F1'] = non_overlapping_windows_coref_metrics_df['CONLL F1'].apply(
        lambda x: f"{x:.2%}")

    non_overlapping_windows_coref_metrics_table = tabulate(non_overlapping_windows_coref_metrics_df, headers="keys",
                                                           tablefmt="github")

    full_annotated_sample_coref_metrics_df = full_annotated_sample_coref_metrics_df.sort_values(
        'tokens_span').reset_index(drop=True)
    full_annotated_sample_coref_metrics_df = full_annotated_sample_coref_metrics_df[
        ['tokens_span', 'mentions_count', 'MUC_f1', 'B3_f1', 'CEAFe_f1', 'CONLL_f1']]
    full_annotated_sample_coref_metrics_df.columns = ['Token count', 'Mention count', 'MUC F1', 'B3 F1', 'CEAFe F1',
                                                      'CONLL F1']

    full_annotated_sample_coref_metrics_df['Token count'] = full_annotated_sample_coref_metrics_df['Token count'].apply(
        lambda x: f"{x:,}")
    full_annotated_sample_coref_metrics_df['Mention count'] = full_annotated_sample_coref_metrics_df[
        'Mention count'].apply(lambda x: f"{x:,}")
    full_annotated_sample_coref_metrics_df['MUC F1'] = full_annotated_sample_coref_metrics_df['MUC F1'].apply(
        lambda x: f"{x:.2%}")
    full_annotated_sample_coref_metrics_df['B3 F1'] = full_annotated_sample_coref_metrics_df['B3 F1'].apply(
        lambda x: f"{x:.2%}")
    full_annotated_sample_coref_metrics_df['CEAFe F1'] = full_annotated_sample_coref_metrics_df['CEAFe F1'].apply(
        lambda x: f"{x:.2%}")
    full_annotated_sample_coref_metrics_df['CONLL F1'] = full_annotated_sample_coref_metrics_df['CONLL F1'].apply(
        lambda x: f"{x:.2%}")

    full_annotated_sample_coref_metrics_table = tabulate(full_annotated_sample_coref_metrics_df, headers="keys",
                                                         tablefmt="github")

    return non_overlapping_windows_coref_metrics_table, full_annotated_sample_coref_metrics_table


def generate_coref_model_card_from_LOOCV_directory(coref_trained_model_directory,
                                                   model_evaluation_files="all",
                                                   read_me_file_name="README",
                                                   token_windows_length=[500, 1000, 2000, 5000, 10000, 25000, 50000]):
    final_model_path = os.path.join(coref_trained_model_directory, "final_model")
    with open(final_model_path, "rb") as file:
        trained_model_infos = pickle.load(file)

    trained_model_infos, overall_coreference_resolution_metrics_df, non_overlapping_windows_coref_metrics_df, full_annotated_sample_coref_metrics_df, corpus_infos_table = generate_overall_LOOCV_metrics_df(
        coref_trained_model_directory,
        trained_model_infos,
        model_evaluation_files=model_evaluation_files,
        token_windows_length=token_windows_length)

    non_overlapping_windows_coref_metrics_table, full_annotated_sample_coref_metrics_table = generate_datacard_metrics_tables(
        non_overlapping_windows_coref_metrics_df, full_annotated_sample_coref_metrics_df)

    foundation_model = trained_model_infos["base_model_name"]
    _, model = load_tokenizer_and_embedding_model(model_name=foundation_model)
    embedding_dim = model.config.hidden_size

    entity_types = trained_model_infos["entity_types"]
    all_files = trained_model_infos["all_files"]
    train_with_validation_ratio = trained_model_infos["train_with_validation_ratio"]
    batch_size = trained_model_infos["batch_size"]
    learning_rate = trained_model_infos["learning_rate"]
    focal_loss_gamma = trained_model_infos["focal_loss_gamma"]
    focal_loss_alpha = trained_model_infos["focal_loss_alpha"]
    pronoun_antecedent_max_distance = trained_model_infos["pronoun_antecedent_max_distance"]
    proper_common_nouns_antecedent_max_distance = trained_model_infos["proper_common_nouns_antecedent_max_distance"]

    model = trained_model_infos["model"]
    model_input_dimension = model.network[0].in_features

    features = trained_model_infos["features"]
    additional_mentions_features_dict = {
        "mention_len": "Length of mentions",
        "start_token_ID_within_sentence": "Position of the mention's start token within the sentence",
        "prop": "Grammatical category of the mentions (pronoun, common noun, proper noun)",
        "head_dependency_relation": "Dependency relation of the mention's head (one-hot encoded)",
        "gender": "Gender of the mentions (one-hot encoded)",
        "number": "Number (singular/plural) of the mentions (one-hot encoded)",
        "grammatical_person": "Grammatical person of the mentions (one-hot encoded)", }

    additional_mention_pairs_features_dict = {
        "mention_ID_delta": "Distance between mention IDs",
        "start_token_delta": "Distance between start tokens of mentions",
        "end_token_delta": "Distance between end tokens of mentions",
        "sentence_ID_delta": "Distance between sentences containing mentions",
        "paragraph_ID_delta": "Distance between paragraphs containing mentions",
        "out_to_in_nested_level_delta": "Difference in nesting levels of mentions",
        "shared_token_ratio": "Ratio of shared tokens between mentions",
        "text_match": "Exact text match between mentions (binary)",
        "head_text_match": "Exact match of mention heads (binary)",
        "syntactic_head_match": "Match of syntactic heads between mentions (binary)",
        "cat_match": "Match of entity types between mentions (binary)",
    }
    additional_mentions_features = [additional_mentions_features_dict[key] for key in
                                    additional_mentions_features_dict.keys() if key in features]
    formatted_additional_mentions_features = " None"
    if len(additional_mentions_features) > 0:
        formatted_additional_mentions_features = '\n  - ' + '\n  - '.join(additional_mentions_features)
    additional_mention_pairs_features = [additional_mention_pairs_features_dict[key] for key in
                                         additional_mention_pairs_features_dict.keys() if key in features]
    formatted_additional_mention_pairs_features = " None"
    if len(additional_mention_pairs_features) > 0:
        formatted_additional_mention_pairs_features = '\n  - ' + '\n  - '.join(additional_mention_pairs_features)

    layers_number = trained_model_infos["layers_number"]
    layers_units = trained_model_infos["layers_units"]
    layer_type = trained_model_infos["layer_type"]
    dropout = trained_model_infos["dropout"]

    read_me = f"""---
language: fr
tags:
- coreference-resolution
- anaphora-resolution
- mentions-linking
- literary-texts
- camembert
- literary-texts
- nested-entities
- propp-fr
license: apache-2.0
metrics:
- MUC
- B3
- CEAF
- CoNLL-F1
base_model:
- {foundation_model}
---

## INTRODUCTION:
This model, developed as part of the [propp-fr project](https://lattice-8094.github.io/propp/), is a **coreference resolution model** built on top of [{foundation_model.split("/")[-1]}](https://huggingface.co/{foundation_model}) embeddings. It is trained to link mentions of the same entity across a text, focusing on literary works in French.

This specific model has been trained to link entities of the following types: {', '.join(entity_types)}.

## MODEL PERFORMANCES (LOOCV):
Overall Coreference Resolution Performances for non-overlapping windows of different length:
{non_overlapping_windows_coref_metrics_table}

Coreference Resolution Performances on the fully annotated sample for each document:
{full_annotated_sample_coref_metrics_table}

## TRAINING PARAMETERS:
- Entities types: {', '.join(entity_types)}
- Split strategy: Leave-one-out cross-validation ({len(all_files)} files)
- Train/Validation split: {train_with_validation_ratio} / {(1 - train_with_validation_ratio):.2f}
- Batch size: {batch_size:,}
- Initial learning rate: {learning_rate}
- Focal loss gamma: {focal_loss_gamma}
- Focal loss alpha: {focal_loss_alpha}
- Pronoun lookup antecedents: {pronoun_antecedent_max_distance}
- Common and Proper nouns lookup antecedents: {proper_common_nouns_antecedent_max_distance}

## MODEL ARCHITECTURE:
Model Input: {model_input_dimension:,} dimensions vector
- Concatenated maximum context {foundation_model.split("/")[-1]} embeddings (2 * {embedding_dim:,} = {2 * embedding_dim:,} dimensions)
- Additional mentions features ({model_input_dimension - 2 * embedding_dim - len(additional_mention_pairs_features):,} dimensions):{formatted_additional_mentions_features}
- Additional mention pairs features ({len(additional_mention_pairs_features)} dimensions):{formatted_additional_mention_pairs_features}

- Hidden Layers:
  - Number of layers: {layers_number}
  - Units per layer: {layers_units:,} nodes
  - Activation function: {layer_type}
  - Dropout rate: {dropout}

- Final Layer:
  - Type: Linear
  - Input: {layers_units} dimensions
  - Output: 1 dimension (mention pair coreference score)

Model Output: Continuous prediction between 0 (not coreferent) and 1 (coreferent) indicating the degree of confidence.

## HOW TO USE:
[Propp Documentation](https://lattice-8094.github.io/propp/quick_start/)

## TRAINING CORPUS:
{corpus_infos_table}

## CONTACT:
mail: antoine [dot] bourgois [at] protonmail [dot] com
"""

    ## Save the model card as a Markdown file
    output_path = os.path.join(coref_trained_model_directory, f"{read_me_file_name}.md")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(read_me)

    print(f"model card {read_me_file_name}.md file has been saved at {output_path}")


# Inference functions
## Mentions Detection Model Inference
def load_coreference_resolution_model(
        model_path="AntoineBourgois/propp-fr_coreference-resolution_camembert-large_PER", force_download=False):
    """
    Loads a coreference resolution model from a specified path. It first checks for the model locally,
    and if not found (or if force_download is True), downloads it from HuggingFace.

    Args:
        model_path (str): Path to the model. This can be a local file, directory, or a HuggingFace model name.
        force_download (bool): If True, the coreference resolution model is downloaded from HuggingFace
                               and the local model is overwritten.

    Returns:
        object: The loaded coreference resolution model.

    Raises:
        requests.exceptions.RequestException: For HTTP errors while downloading the model.
        pickle.UnpicklingError: For errors during model deserialization.
        FileNotFoundError: If the specified local file or directory doesn't exist.
        Exception: For other unexpected errors.
    """

    def download_model_from_huggingface():
        """Downloads the model from HuggingFace and saves it locally."""
        print(f"Downloading model from HuggingFace: https://huggingface.co/{model_path}")
        url_model_path = f"https://huggingface.co/{model_path}/resolve/main/final_model"

        try:
            response = requests.get(url_model_path)
            response.raise_for_status()  # Raise an exception for HTTP errors
            coreference_resolution_model = pickle.loads(response.content)  # Deserialize the downloaded model
            print("Model Downloaded Successfully")

            # Save the model locally for future use
            mentions_models_directory = os.path.dirname(local_coreference_resolution_model_path)
            absolute_directory = os.path.abspath(mentions_models_directory)
            if not os.path.exists(absolute_directory):
                os.makedirs(absolute_directory)

            print(f"Saving model locally to: {absolute_directory}")
            with open(local_coreference_resolution_model_path, "wb") as file:
                pickle.dump(coreference_resolution_model, file)

            return coreference_resolution_model

        except requests.exceptions.RequestException as req_err:
            print(f"Error downloading model from HuggingFace: {req_err}")
            raise
        except pickle.UnpicklingError as pickle_err:
            print(f"Error unpickling the model: {pickle_err}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
            raise

    # Local model path to check
    current_directory = os.getcwd()
    local_coreference_resolution_model_path = os.path.join(current_directory, f"{model_path}/final_model")

    try:
        if not force_download:
            # 1. Attempt to load model from the given file path
            if os.path.isfile(model_path):  # Check if the provided path is a file
                with open(model_path, "rb") as file:
                    coreference_resolution_model = pickle.load(file)
                print(f"Model Loaded Successfully from local path: {model_path}")
                return coreference_resolution_model

            # 2. Attempt to load model from the "final_model.pkl" in the current working directory
            if os.path.exists(local_coreference_resolution_model_path):
                with open(local_coreference_resolution_model_path, "rb") as file:
                    coreference_resolution_model = pickle.load(file)
                print(f"Model Loaded Successfully from local path: {local_coreference_resolution_model_path}")
                return coreference_resolution_model

        # 3. If force_download is True or local model is not found, download the model
        return download_model_from_huggingface()

    except FileNotFoundError as file_err:
        print(f"File not found: {file_err}")
        raise
    except pickle.UnpicklingError as pickle_err:
        print(f"Error unpickling the model: {pickle_err}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def generate_clusters(matrix, verbose=0):
    """
    Generate clusters from the predicted coreference matrix using an iterative DFS approach.

    Args:
    - matrix (np.ndarray): Square matrix where 1 indicates coreference, and -1 indicates no coreference.

    Returns:
    - clusters (list of lists): A list of clusters, where each cluster is a list of indices.
    """
    N = len(matrix)
    clusters = []
    processed_mentions = set()  # Use set for O(1) average time complexity for membership checks

    if verbose == 1:
        for i in tqdm(range(N), desc="Clustering Coreferential Mentions", leave=False):
            if i not in processed_mentions:
                coref_ids = np.where(matrix[i, :] == 1)[0]
                clusters.append(list(coref_ids))
                processed_mentions.update(coref_ids)  # Use update to add all coref_ids at once
    elif verbose == 0:
        for i in range(N):
            if i not in processed_mentions:
                coref_ids = np.where(matrix[i, :] == 1)[0]
                clusters.append(list(coref_ids))
                processed_mentions.update(coref_ids)  # Use update to add all coref_ids at once
    # Sort clusters by their length in descending order
    clusters = sorted(clusters, key=len, reverse=True)

    return clusters


# def assign_coreference_to_proper_mentions_if_A_is_part_of_B(entities_df):
#     """
#     Assign coreference predictions to proper mentions where one mention is part of another.
#
#     Parameters:
#         entities_df (pd.DataFrame): DataFrame containing entity mentions with attributes such as
#                                     'text', 'prop', 'gender', and 'number'.
#
#     Returns:
#         pd.DataFrame: DataFrame containing pairs of indices (`A`, `B`) that are predicted to be coreferent,
#                       along with the `coreference_prediction` column set to 1.
#     """
#     # Create a copy and preprocess text
#     filtered_df = entities_df.copy()
#     filtered_df['index'] = filtered_df.index
#     filtered_df['text'] = filtered_df['text'].str.lower()
#
#     # Nominal prefixes for filtering proper nouns
#     nominal_starts_with_list = ['monsieur', 'm.', 'madame', 'mme', 'mademoiselle', 'mon ami', 'mon amie', 'mon cher',
#                                 'ma chère', 'cher', 'chère', 'maître', 'le malheureux', 'malheureux']
#
#     # Filter for proper nouns ('PROP') or specific nominal forms ('NOM')
#     filtered_df = filtered_df[
#         (filtered_df['prop'] == 'PROP') |
#         ((filtered_df['prop'] == 'NOM') & (
#             filtered_df['text'].str.startswith(tuple(f"{prefix} " for prefix in nominal_starts_with_list))))
#         ]
#
#     # Filter based on gender and number
#     filtered_df = filtered_df[filtered_df['gender'].isin(['Male', 'Female', 'Not_Assigned'])]
#     filtered_df = filtered_df[filtered_df['number'].isin(['Singular', 'Plural'])]
#
#     # Perform self-merge to create candidate pairs with matching gender and number
#     merged_df = pd.merge(
#         filtered_df, filtered_df, on=['gender', 'number'], suffixes=('_A', '_B')
#     )
#
#     # Ensure at least one of the mentions in the pair is proper ('PROP')
#     merged_df = merged_df[(merged_df['prop_A'] == 'PROP') | (merged_df['prop_B'] == 'PROP')]
#
#     # Avoid self-matches by ensuring index_A < index_B
#     merged_df = merged_df[merged_df['index_A'] < merged_df['index_B']]
#
#     # Helper function to check full token containment
#     def is_full_token_containment(row):
#         """
#         Check if one mention's text is a full token contained in the other's text.
#         """
#         text_a = row['text_A']
#         text_b = row['text_B']
#         pattern_a = r'\b' + re.escape(text_a) + r'\b'
#         pattern_b = r'\b' + re.escape(text_b) + r'\b'
#         return bool(re.search(pattern_a, text_b)) or bool(re.search(pattern_b, text_a))
#
#     # Apply containment check
#     condition = merged_df.apply(is_full_token_containment, axis=1)
#
#     # Filter pairs based on containment
#     processed_mention_pairs = merged_df[condition].copy()
#
#     # Add coreference prediction
#     processed_mention_pairs['coreference_prediction'] = 1
#
#     # Select and rename columns for output
#     result_df = processed_mention_pairs[['index_A', 'index_B', 'coreference_prediction']]
#     result_df.columns = ['A', 'B', 'coreference_prediction']
#
#     return result_df

def assign_coreference_to_proper_mentions_if_A_is_part_of_B(entities_df):
    """
    Assign coreference predictions to proper mentions where one mention is part of another.

    Parameters:
        entities_df (pd.DataFrame): DataFrame containing entity mentions with attributes such as
                                    'text', 'prop', 'gender', and 'number'.

    Returns:
        pd.DataFrame: DataFrame containing pairs of indices (`A`, `B`) that are predicted to be coreferent.
    """
    # Convert text to lowercase and store index
    entities_df = entities_df.copy()
    entities_df['text'] = entities_df['text'].str.lower()
    entities_df['index'] = entities_df.index

    # Define nominal prefixes
    nominal_starts_with = ('monsieur', 'm.', 'madame', 'mme', 'mademoiselle', 'mon ami', 'mon amie',
                           'mon cher', 'ma chère', 'cher', 'chère', 'maître', 'le malheureux', 'malheureux')

    # Filter proper mentions
    mask_proper = (entities_df['prop'] == 'PROP') | ((entities_df['prop'] == 'NOM') & entities_df['text'].str.startswith(nominal_starts_with))
    filtered_df = entities_df[mask_proper]

    # Filter by gender and number
    filtered_df = filtered_df[filtered_df['gender'].isin(['Male', 'Female', 'Not_Assigned'])]
    filtered_df = filtered_df[filtered_df['number'].isin(['Singular', 'Plural'])]

    # Create mention list to avoid memory-intensive self-merge
    mention_list = filtered_df[['index', 'text', 'gender', 'number', 'prop']].values

    results = []
    for i, (idx_A, text_A, gender_A, num_A, prop_A) in enumerate(mention_list):
        for j, (idx_B, text_B, gender_B, num_B, prop_B) in enumerate(mention_list):
            if idx_A >= idx_B:
                continue  # Avoid self-matching and duplicates

            if prop_A != 'PROP' and prop_B != 'PROP':
                continue  # Ensure at least one mention is proper

            # Use in-string containment check without regex for efficiency
            if text_A in text_B or text_B in text_A:
                results.append((idx_A, idx_B, 1))

    # Convert results to DataFrame
    return pd.DataFrame(results, columns=['A', 'B', 'coreference_prediction'])

def assign_coreference_to_direct_nested_mentions(CAT_entities_df):
    """
    Assign coreference predictions to direct nested mentions.

    Parameters:
        CAT_entities_df (pd.DataFrame): DataFrame containing entity mentions with attributes such as
                                        'start_token', 'end_token', 'nested_entities_count', and 'out_to_in_nested_level'.

    Returns:
        pd.DataFrame: DataFrame containing pairs of indices (`A`, `B`) with `coreference_prediction` set to 0.
    """
    processed_mention_pairs = []
    # Convert DataFrame to list of dicts for easier access
    entities = CAT_entities_df.to_dict(orient='records')

    # Filter entities to only those with nested entities
    outer_entities_df = CAT_entities_df[
        (CAT_entities_df['nested_entities_count'] > 0)
        & (~CAT_entities_df['text'].str.lower().str.contains('|'.join(['qui', 'comme', 'dont', 'que'])))
        ].sort_values('nested_entities_count', ascending=False)

    # Iterate through outer entities
    for i in outer_entities_df.index:
        # Get the outer entity's properties
        i_start, i_end, i_outer_to_inner_nested_level = entities[i]["start_token"], entities[i]["end_token"], \
        entities[i]["out_to_in_nested_level"]

        # Find nested entities within the bounds of the outer entity
        nested_entities = CAT_entities_df[
            (CAT_entities_df.index != i)  # Exclude the current entity itself
            & (CAT_entities_df['start_token'] >= i_start)
            & (CAT_entities_df['end_token'] <= i_end)
            & (CAT_entities_df[
                   'out_to_in_nested_level'] == i_outer_to_inner_nested_level + 1)]  # Only apply to nested entities of nested direct lower nested level

        # Create mention pairs for the nested entities with the outer entity
        for j in nested_entities.index:
            processed_mention_pairs.append({"A": i, "B": j, "coreference_prediction": 0})

        # If there are multiple nested entities, generate mention pairs between them
        if len(nested_entities) > 1:
            nested_entities_pairs = list(combinations(nested_entities.index, 2))
            for A, B in nested_entities_pairs:
                processed_mention_pairs.append({"A": A, "B": B, "coreference_prediction": 0})
    # Convert to DataFrame
    processed_mention_pairs = pd.DataFrame(processed_mention_pairs)

    return processed_mention_pairs
def assign_coreference_to_explicit_proper_and_noun_phrase_mentions(entities_df):
    """
    Assign coreference predictions to explicit proper noun and noun phrase mentions based on linguistic patterns.

    Parameters:
        entities_df (pd.DataFrame): DataFrame containing entity mentions with attributes such as 'prop', 'mention_len', and 'text'.

    Returns:
        pd.DataFrame: DataFrame containing pairs of indices (`A`, `B`) with `coreference_prediction` set to 1.
    """
    filtered_df = entities_df.copy()
    filtered_df['index'] = filtered_df.index
    filtered_df['text'] = filtered_df['text'].str.lower()

    # Filter for noun phrases with a minimum length of 2
    filtered_df = filtered_df[(filtered_df['prop'] == 'NOM') & (filtered_df['mention_len'] >= 2)]

    # Dictionary for replacing equivalent terms with a common key
    replacing_dict = {'monsieur': ['monsieur', 'm.', 'sieur', 'le sieur', 'du sieur'],
                      'madame': ['madame', 'mme', 'mme.'],
                      'mademoiselle': ['mademoiselle', 'mlle', 'mme.'],
                      'lord': ['lord', 'le lord', 'du lord'],
                      'chevalier': ['chevalier', 'le chevalier', 'du chevalier'],
                      'capitaine': ['capitaine', 'le capitaine', 'du capitaine'],
                      'cardinal': ['cardinal', 'le cardinal', 'du cardinal'],
                      'curé': ['curé', 'le curé', 'du curé'],
                      'marquis': ['marquis', 'le marquis', 'du marquis'],
                      'baron': ['baron', 'le baron', 'du baron', 'au baron'],
                      'comte': ['comte', 'le comte', 'du comte'],
                      'lieutenant': ['lieutenant', 'le lieutenant', 'du lieutenant'],
                      }

    # Replace equivalent terms with a common key
    def replace_by_dict(text, replacing_dict):
        for key, values in replacing_dict.items():
            for value in values:
                # Create a pattern that checks for start of string or space before and space or end of string after
                pattern = r'(^|\s)' + re.escape(value) + r'(\s|$)'
                replacement = r'\1' + key + r'\2'
                text = re.sub(pattern, replacement, text)
        return text

    filtered_df['text'] = filtered_df['text'].apply(lambda x: replace_by_dict(x, replacing_dict))

    # Define valid phrases and prefixes for matching
    valid_phrases = ['la vierge', 'le diable', 'du diable', 'les jumeaux']
    starts_with_prefixes = {
        'baron ', 'capitaine ', 'cardinal ', 'chevalier ', 'comte ', 'curé ', 'dame ', 'de la baronne ', 'donna ',
        'du baron ', 'du cardinal ', "l' abbé", "l' ancien ", "l' ancienne ", "l' empereur", "l' évêque", 'la baronne ',
        'la comtesse ', 'la dauphine', 'la donna ', 'la duchesse ', 'la famille ', 'la marquise ', 'la mère ',
        'la reine ', 'la société ', 'le baron ', 'le cardinal ', 'le commandant ', 'le comte ', 'le curé ',
        'le dauphin ', 'le docteur ', 'le duc ', 'le lieutenant ', 'le lord ', 'le marquis ', 'le prêtre ', 'le père ',
        'le roi ', 'le vieux ', 'lieutenant ', 'lord ', 'madame ', 'mademoiselle ', 'marquis ', 'maître ', 'monsieur ',
        'mère ', 'oncle ', 'père ', 'sa fille ', 'son fils ', 'tante '
    }

    # Filter entities based on valid phrases or prefixes
    filtered_df = filtered_df[
        filtered_df['text'].isin(valid_phrases) |
        filtered_df['text'].str.startswith(tuple(starts_with_prefixes))
        ]

    # Self-merge for efficient matching
    merged_df = pd.merge(filtered_df, filtered_df, on='text', suffixes=('_A', '_B'))

    # Remove self-matches where A and B are the same
    processed_mention_pairs = merged_df[merged_df['index_A'] < merged_df['index_B']].copy()

    # Add coreference prediction column
    processed_mention_pairs['coreference_prediction'] = 1

    # Selecting only relevant columns for output
    result_df = processed_mention_pairs[['index_A', 'index_B', 'coreference_prediction']]
    result_df.columns = ['A', 'B', 'coreference_prediction']

    return result_df


def rank_predicted_antecedents(predicted_mention_pairs):
    predicted_mention_pairs = predicted_mention_pairs.copy()  # Avoid modifying in place
    predicted_mention_pairs["ranked_antecedent"] = 0  # Initialize column

    # Filter only rows where coreference_prediction == 1
    coreferent_df = predicted_mention_pairs[predicted_mention_pairs["coreference_prediction"] == 1]

    # Sort by 'B' and 'confidence' in descending order
    coreferent_df = coreferent_df.sort_values(["B", "confidence"], ascending=[True, False])

    # Compute ranking per group using 'cumcount()' (which is vectorized)
    coreferent_df["ranked_antecedent"] = coreferent_df.groupby("B").cumcount() + 1

    # Update the original DataFrame in a single operation
    predicted_mention_pairs.loc[coreferent_df.index, "ranked_antecedent"] = coreferent_df["ranked_antecedent"]

    return predicted_mention_pairs
def harmonize_prefix(entities_df, replace_dict=None, text_column = "text"):
    if replace_dict == None:
        replace_dict = {"mme": "madame",
                        "m.": "monsieur",
                        "mgr": "monsieur",
                        "sieur": "monsieur",
                        "le sieur": "monsieur",
                        "du sieur": "monsieur",
                        "mm.": "messieurs",
                        "me": "maître",
                        "dr": "docteur",
                        "le dr": "docteur",
                        "du dr": "docteur",
                        "l' abbé": "abbé",
                        "le marquis": "marquis",
                        "la marquise": "marquise",
                        "le comte": "comte",
                        "du comte": "comte",
                        "la comtesse": "comtesse",
                        "le duc": "duc",
                        "du duc": "duc",
                        "le curé": "curé",
                        "la duchesse": "duchesse",
                        "la baronne": "baronne",
                        "le baron": "baron",
                        "du baron": "baron",
                        "au baron": "baron",
                        "le capitaine": "capitaine",
                        "du capitaine": "capitaine",
                        "le lieutenant": "lieutenant",
                        "du lieutenant": "lieutenant",
                        "le cardinal": "cardinal",
                        "le docteur": "docteur",
                        "du docteur": "docteur",
                        "du colonel": "colonel",
                        "le colonel": "colonel",
                        "le chevalier": "chevalier",
                        "mlle": "mademoiselle",
                        "l' ambassadeur": "ambassadeur",
                        "le sous-lieutenant": "sous-lieutenant",
                        "du sous-lieutenant": "sous-lieutenant",
                        "le père": "père",
                        "la mère": "mère",
                        "le roi": "roi",
                        "au roi": "roi",
                        "la reine": "reine",
                        "la donna": "donna",
                        "aux": "les",
                        "au": "le",
                        "ma cousine": "cousine",
                        "la famille": "famille",
                        "la société": "société",
                        "la maison": "maison",
                        }

    # Function to replace values from the dictionary
    def replace_from_dict(text):
        # Now replace other values from the dictionary
        for key, value in replace_dict.items():
            if text.startswith(key):
                text = text.replace(f"{key} ", f"{value} ").strip()
        return text

    # Apply the replacement function
    entities_df[text_column] = entities_df[text_column].fillna("").apply(replace_from_dict)
    return entities_df
def strip_proper_mentions(df):


    prefix_to_remove = ["le pauvre", "ce pauvre", "ce digne", 'le pauvre', "mon pauvre", "la pauvre", "ma pauvre", "pauvre", 'mon cher', 'mon bon', 'pauvre', 'ce bon', 'bon', "bonne", 'digne', "la perfide", "le bon", "chère", "l' infidèle", "l' ingrate", "la perfide", "ma chère", 'chère', "perfide", "belle", "l' indigne", "le malheureux", "la malheureuse", "l' heureuse", "heureuse", "la pauvre", "l' orgueilleux", "l' énergique", "cette pauvre", "fidèle", 'loyale', 'la loyale', 'ma loyale', 'la pale', 'le beau', "cher", "sir", 'la belle', 'la fidèle', 'la douce', "le flegmatique", "la pâle", "la triste", "cette", "ma bonne", "la bonne", "la curieuse", "la plaintive", "", "énigmatique", "un certain", "une certaine", "espiègle", "la simple", "la timide", "la trop naïve", "la naïve", "malheureux", "la sage", "son cher", 'ignorante', "l' ignorante", "du pauvre", "la toute puissante", "le tout puissant", "toute-puissante", "tout-puissant", "une pareille", "un pareil", "le sage", "le cher", "sotte", "la sotte", "le sot", "du prudent", "le prudent", "l' imprudent", "imprudent", "imprudente", "le farouche", "le terrible", "le généreux", "le pieu", "l' imprudente", "la prudente", "l' honnête", "honnête", "cet", 'ta', 'ma', 'mon', 'ton', 'son', 'sa', 'ce', "l'", "notre", 'ton', 'au', "à", "un", "une", "le", "votre", "notre"]
    prefix_to_remove = [f"{prefix} " for prefix in prefix_to_remove]
    # Create regex pattern for prefixes
    prefix_pattern = r"^(" + "|".join(map(re.escape, prefix_to_remove)) + ")"


    # List of suffixes to remove
    suffix_to_remove = ["lui-même", "elle-même", ".", "chérie", "déja décrit", "ici présent", "ici présente"]
    suffix_to_remove = [f" {sufix}" for sufix in suffix_to_remove]

    # Pattern to remove anything after "qu'" or "qui"
    qui_pattern = r"(qu'|qui)\s.*"

    # Create regex pattern for suffixes
    suffix_pattern = r"(" + "|".join(map(re.escape, suffix_to_remove)) + r")$"
    suffix_pattern = rf"{qui_pattern}|{suffix_pattern}"

    df['text'] = df['text'].str.replace(prefix_pattern, "", regex=True)
    df['text'] = df['text'].str.replace(suffix_pattern, "", regex=True)
    df['text'] = df['text'].str.strip()
    return df
def get_harmonize_PROP_entities_df(entities_df):
    PROP_entities_df = entities_df[entities_df["prop"] == "PROP"].copy()
    PROP_entities_df = harmonize_prefix(PROP_entities_df)
    PROP_entities_df = strip_proper_mentions(PROP_entities_df)
    PROP_entities_df["mention_len"] = PROP_entities_df["text"].str.split().apply(len)

    return PROP_entities_df
def get_grouped_proper_mentions_df(df):
    df["count"] = 1
    if not "file_name" in df.columns:
        df["file_name"] = ""
    grouped_PROP_entities_df = df.groupby(
        ['text', 'number', 'gender', 'mention_len'],
        as_index=False  # Keeps the grouped columns as regular columns instead of indexes
    ).apply(
        lambda group: pd.Series({
            'count': group['count'].sum(),
            'row_ids': group.index.tolist()  # Store row indices as a list
        }),
        include_groups=False  # Exclude group columns from the operation
    )
    grouped_PROP_entities_df = grouped_PROP_entities_df[grouped_PROP_entities_df["count"] > 1]

    return grouped_PROP_entities_df
def get_gender_number_incompatible_proper_pairs_df(df):
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df["number"].isin(["Plural", "Singular"]))
                                            | (filtered_df["number"].isin(["Plural", "Singular"]))]
    filtered_df = filtered_df.sort_values(["mention_len", "count"], ascending=[True, False])

    non_coreferent_pairs = set()
    for text, number, gender, count in filtered_df[["text", "number", "gender", "count"]].values:
        # target_df = filtered_df[filtered_df["text"].str.contains(rf'\b{text}\b', regex=True, na=False)]
        target_df = filtered_df[filtered_df["text"].str.contains(rf'\b{re.escape(text)}\b', regex=True, na=False)]
        male_target_df = target_df[target_df["gender"] == "Male"]
        female_target_df = target_df[target_df["gender"] == "Female"]
        singular_target_df = target_df[target_df["number"] == "Singular"]
        plural_target_df = target_df[target_df["number"] == "Plural"]

        gender_incompatible_pairs = list(product(male_target_df["row_ids"].tolist(), female_target_df["row_ids"].tolist()))
        number_incompatible_pairs = list(product(singular_target_df["row_ids"].tolist(), plural_target_df["row_ids"].tolist()))

        for ids_list in gender_incompatible_pairs + number_incompatible_pairs:
            for mention_pair in product(*ids_list):
                non_coreferent_pairs.add(tuple(sorted(mention_pair)))

    non_coreferent_pairs_df = pd.DataFrame(sorted(non_coreferent_pairs), columns=["A", "B"])
    non_coreferent_pairs_df["confidence"] = 1
    non_coreferent_pairs_df["coreference_prediction"] = 0

    return non_coreferent_pairs_df
def get_conjunction_incompatible_proper_pairs_df(df):
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df["number"] == "Plural"]
    filtered_df = filtered_df[filtered_df["text"].str.contains(r" et |, ", regex=True, na=False)]
    incompatible_mention_lists = filtered_df["text"].tolist()
    incompatible_mention_lists = [list(set(mention.split(" et "))) for mention in incompatible_mention_lists]

    non_coreferent_pairs = set()
    for incompatible_mention_set in incompatible_mention_lists:
        mention_ids = []
        for mention in incompatible_mention_set:
            mention_ids.append(df[df["text"] == mention].index.tolist())
        # Generate all possible pairs
        pairs = list(product(*mention_ids))
        for pair in pairs:
            non_coreferent_pairs.add(tuple(sorted(pair)))

    non_coreferent_mention_pairs = set()
    for pair in non_coreferent_pairs:
        if len(pair) != 2:
            continue
        A_ids, B_ids = pair
        A_ids = df.loc[A_ids, "row_ids"]
        B_ids = df.loc[B_ids, "row_ids"]
        for pair in product(A_ids, B_ids):
            non_coreferent_mention_pairs.add(tuple(sorted(pair)))

    non_coreferent_pairs_df = pd.DataFrame(sorted(non_coreferent_mention_pairs), columns=["A", "B"])
    non_coreferent_pairs_df["confidence"] = 1
    non_coreferent_pairs_df["coreference_prediction"] = 0

    return non_coreferent_pairs_df
def get_propagated_proper_mentions_pairs(grouped_PROP_entities_df,
                                         predicted_mention_pairs,
                                         verbose=1,
                                         confidence=1,
                                         homogeneity_rate_threshold=1/3):

    grouped_PROP_entities_df = grouped_PROP_entities_df[grouped_PROP_entities_df["count"] >= 5].copy()
    # Prepare the 'A' and 'B' combinations for pairs
    mention_ids_list = grouped_PROP_entities_df["row_ids"].tolist()
    ids_pairs = list(product(mention_ids_list, repeat=2))

    # Precompute mention pair keys for faster lookups
    predicted_mention_pairs['pair_key'] = predicted_mention_pairs.apply(
        lambda x: frozenset([x["A"], x["B"]]), axis=1
    )

    # Merge self with itself to find pairs efficiently
    # The idea here is to create a DataFrame where each pair of mentions (A, B) is represented
    # and we can use the merge operation to find intersections between these pairs.
    merged_pairs = []
    if verbose == 0:
        for A_ids, B_ids in ids_pairs:
            # Convert list of A_ids and B_ids to sets for faster lookup
            A_ids_set = set(A_ids)
            B_ids_set = set(B_ids)

            # Check using set intersection for faster lookups
            filtered_pairs = predicted_mention_pairs[
                predicted_mention_pairs['A'].isin(A_ids_set) & predicted_mention_pairs['B'].isin(B_ids_set)
                | predicted_mention_pairs['A'].isin(B_ids_set) & predicted_mention_pairs['B'].isin(A_ids_set)
                ]

            if not filtered_pairs.empty:
                # Calculate necessary statistics for the pair
                pairs_dict = {
                    "A_text": filtered_pairs.iloc[0]["A_text"],
                    "B_text": filtered_pairs.iloc[0]["B_text"],
                    "all_pairs_count": int(0.5 * len(A_ids) * len(B_ids)),
                    "scored_pairs_count": len(filtered_pairs),
                    "average_prediction": filtered_pairs["model_prediction"].mean(),
                    "average_confidence": filtered_pairs["confidence"].mean(),
                    "average_coreference_prediction": filtered_pairs["coreference_prediction"].mean(),
                    "homogeneity_rate": abs(filtered_pairs["coreference_prediction"].mean() - 0.5) * 2,
                    "A_ids": A_ids,
                    "B_ids": B_ids,
                }
                merged_pairs.append(pairs_dict)
    elif verbose == 1:
        for A_ids, B_ids in tqdm(ids_pairs, desc="Propagating Coref Decision to Global Scale", leave=False):
            # Convert list of A_ids and B_ids to sets for faster lookup
            A_ids_set = set(A_ids)
            B_ids_set = set(B_ids)

            # Check using set intersection for faster lookups
            filtered_pairs = predicted_mention_pairs[
                predicted_mention_pairs['A'].isin(A_ids_set) & predicted_mention_pairs['B'].isin(B_ids_set)
                | predicted_mention_pairs['A'].isin(B_ids_set) & predicted_mention_pairs['B'].isin(A_ids_set)
            ]

            if not filtered_pairs.empty:
                # Calculate necessary statistics for the pair
                pairs_dict = {
                    "A_text": filtered_pairs.iloc[0]["A_text"],
                    "B_text": filtered_pairs.iloc[0]["B_text"],
                    "all_pairs_count": int(0.5 * len(A_ids) * len(B_ids)),
                    "scored_pairs_count": len(filtered_pairs),
                    "average_prediction": filtered_pairs["model_prediction"].mean(),
                    "average_confidence": filtered_pairs["confidence"].mean(),
                    "average_coreference_prediction": filtered_pairs["coreference_prediction"].mean(),
                    "homogeneity_rate": abs(filtered_pairs["coreference_prediction"].mean() - 0.5) * 2,
                    "A_ids": A_ids,
                    "B_ids": B_ids,
                }
                merged_pairs.append(pairs_dict)

    if not merged_pairs:
        return pd.DataFrame()

    # Now, convert merged_pairs to a DataFrame
    propagate_pairs_df = pd.DataFrame(merged_pairs)

    # Drop duplicates based on key columns
    propagate_pairs_df.drop_duplicates(subset=["A_text", "B_text", 'all_pairs_count', "average_prediction"], inplace=True)

    # Sort by homogeneity_rate and scored_pairs_count
    propagate_pairs_df = propagate_pairs_df.sort_values(by=["homogeneity_rate", "scored_pairs_count"], ascending=[False, False])

    # Apply the homogeneity_rate threshold filter
    propagate_pairs_df = propagate_pairs_df[propagate_pairs_df["homogeneity_rate"] > homogeneity_rate_threshold]

    # Compute the coreference prediction based on average values
    propagate_pairs_df["coreference_prediction"] = (propagate_pairs_df["average_coreference_prediction"] > 0.5).astype(int)

    # Prepare final mention pairs from the dataframe
    mention_pairs = [
        {"A": A, "B": B, "coreference_prediction": coreference_prediction, "confidence": confidence}
        for A_ids, B_ids, coreference_prediction in propagate_pairs_df[["A_ids", "B_ids", "coreference_prediction"]].values
        for A, B in product(A_ids, B_ids)
    ]

    return pd.DataFrame(mention_pairs)
def rule_based_postprocessing(CAT_entities_df):
    post_processed_mention_pairs = []
    rule_based_postprocess_pairs_df = pd.DataFrame()
        # Apply additional rules based on the provided configuration
    post_processed_mention_pairs.append(assign_coreference_to_proper_mentions_if_A_is_part_of_B(CAT_entities_df))
    post_processed_mention_pairs.append(assign_coreference_to_direct_nested_mentions(CAT_entities_df))
    post_processed_mention_pairs.append(assign_coreference_to_explicit_proper_and_noun_phrase_mentions(CAT_entities_df))

    if len(post_processed_mention_pairs) != 0:
        rule_based_postprocess_pairs_df = pd.concat(post_processed_mention_pairs)

    return rule_based_postprocess_pairs_df

def postprocess_mentions_pairs(predicted_mention_pairs,
                               CAT_entities_df=None,
                               propagate_coref=False,
                               rule_based_postprocess=True,
                               verbose=0):

    mention_pairs = predicted_mention_pairs.copy()
    mention_pairs = mention_pairs[mention_pairs["ranked_antecedent"] == 1].sort_values("B", ascending=True).reset_index(drop=True) # highest ranked antecedent only
    mention_pairs = mention_pairs.sort_values("confidence", ascending=False).reset_index(drop=True)

    # Filter entities DataFrame to only include specified entity types
    PROP_entities_df = get_harmonize_PROP_entities_df(CAT_entities_df.copy())

    gender_number_incompatible_pairs_df, conjunction_incompatible_pairs_df, propagated_mention_pairs_df, rule_based_postprocess_pairs_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Initialize DataFrames

    if len(PROP_entities_df) != 0:
        grouped_PROP_entities_df = get_grouped_proper_mentions_df(PROP_entities_df)
        gender_number_incompatible_pairs_df = get_gender_number_incompatible_proper_pairs_df(grouped_PROP_entities_df)
        conjunction_incompatible_pairs_df = get_conjunction_incompatible_proper_pairs_df(grouped_PROP_entities_df)
    #
        if propagate_coref == True:
            propagated_mention_pairs_df = get_propagated_proper_mentions_pairs(grouped_PROP_entities_df, predicted_mention_pairs, verbose=verbose)

    if rule_based_postprocess == True:
        rule_based_postprocess_pairs_df = rule_based_postprocessing(CAT_entities_df)

    # merge mention_pairs
    mention_pairs = pd.concat([gender_number_incompatible_pairs_df,
                               conjunction_incompatible_pairs_df,
                               propagated_mention_pairs_df,
                               rule_based_postprocess_pairs_df,
                               mention_pairs])

    mention_pairs = mention_pairs.drop_duplicates(subset=["A", "B"]).reset_index(drop=True)

    return mention_pairs

def use_characters_alias_list(characters_alias_list, CAT_entities_df, mention_pairs):
    # Accept dict or list
    if isinstance(characters_alias_list, dict):
        # assume dict[str, list[str]] → use only alias lists
        characters_alias_list = list(characters_alias_list.values())

    elif not isinstance(characters_alias_list, list):
        raise TypeError(
            "characters_alias_list must be a dict or a list of lists"
        )
    if not all(isinstance(x, (list, tuple)) for x in characters_alias_list):
        raise ValueError(
            "characters_alias_list must be a list of lists (aliases per character) OR a dict with values being lists"
        )

    characters_mentions_lists = []
    for character_aliases in characters_alias_list:
        mentions_lists = CAT_entities_df[CAT_entities_df["text"].isin(character_aliases)].index.tolist()
        characters_mentions_lists.append(mentions_lists)

    text_pairs = []
    confidence = 1
    for i, A_character_mentions in enumerate(characters_mentions_lists):
        A_character_mentions = sorted(A_character_mentions)
        coreference_prediction = 1
        positive_pairs = [
            (a, b, coreference_prediction, confidence)
            for idx, a in enumerate(A_character_mentions)
            for b in A_character_mentions[idx:]
        ]
        text_pairs.extend(positive_pairs)
        for B_character_mentions in characters_mentions_lists[i+1:]:
            coreference_prediction = 0
            negative_pairs = [(min(a, b), max(a, b), coreference_prediction, confidence)
                              for a, b in product(A_character_mentions, B_character_mentions)]
            text_pairs.extend(negative_pairs)

    alias_mention_pairs_df = pd.DataFrame(text_pairs, columns=["A", "B", "coreference_prediction", "confidence"])
    mention_pairs = pd.concat([alias_mention_pairs_df,
                                   mention_pairs])
    mention_pairs = mention_pairs.drop_duplicates(subset=["A", "B"]).reset_index(drop=True)
    return mention_pairs


def perform_coreference(entities_df=None,
                        tokens_embedding_tensor=None,
                        coreference_resolution_model=None,
                        batch_size=10000,
                        propagate_coref=False,
                        rule_based_postprocess=True,
                        characters_alias_list=None,
                        verbose=1):
    """
    Perform coreference resolution on the given entities DataFrame.

    Args:
    - entities_df (pd.DataFrame): DataFrame containing the entities with their attributes.
    - tokens_embedding_tensor (np.ndarray): Token embeddings tensor for the document.
    - coreference_resolution_model (dict): Dictionary containing coreference model and parameters:
        - "entity_types": List of entity types to consider for coreference resolution.
        - "pronoun_antecedent_max_distance": Max distance for pronoun-antecedent pairs.
        - "proper_common_nouns_antecedent_max_distance": Max distance for proper/common noun antecedents.
        - "features": Features to use for coreference resolution.
        - "model": The trained coreference resolution model.
        - "mention_pairs_post_process": Post-processing configuration for mention pairs.
    - batch_size (int): Batch size for processing mention pairs during inference.

    Returns:
    - pd.DataFrame: Updated entities_df with a "COREF" column indicating cluster IDs.
    """

    # Extract relevant settings and model components
    entity_types = coreference_resolution_model["entity_types"]
    pronoun_antecedent_max_distance = coreference_resolution_model["pronoun_antecedent_max_distance"]
    proper_common_nouns_antecedent_max_distance = coreference_resolution_model[
        "proper_common_nouns_antecedent_max_distance"]
    features = coreference_resolution_model["features"]
    coreference_model = coreference_resolution_model["model"]

    # Filter entities DataFrame to only include specified entity types
    CAT_entities_df = entities_df[entities_df["cat"].isin(entity_types)].copy().reset_index(drop=True)

    # Step 1: Generate mention embeddings for the filtered entities
    mentions_embeddings_tensor = get_mentions_embeddings(CAT_entities_df, tokens_embedding_tensor)
    del tokens_embedding_tensor
    gc.collect()

    # Step 2: Initialize mention pairs DataFrame based on distance constraints
    mention_pairs_df = initialize_mention_pairs_df(CAT_entities_df,
                                                   pronoun_antecedent_max_distance=pronoun_antecedent_max_distance,
                                                   proper_common_nouns_antecedent_max_distance=proper_common_nouns_antecedent_max_distance)

    print("Generating Mention Pairs Features Array...")
    # Step 3: Generate features for mention pairs
    features_array = generate_mention_pairs_features_array(mention_pairs_df,
                                                           CAT_entities_df,
                                                           features=features)
    # Prepare test data for the coreference model
    test_data = {
        "overall_mention_pairs_df": mention_pairs_df.to_numpy(),
        "overall_mentions_embeddings_tensor": mentions_embeddings_tensor,  # NumPy array
        "overall_features_tensor": features_array,  # NumPy array
        "overall_labels_tensor": np.zeros(len(features_array))  # Dummy labels (not used in inference)
    }

    # Step 4: Get predictions for mention pairs
    predictions = get_predictions(coreference_model, test_data, batch_size=batch_size, verbose=verbose)
    del test_data, features_array, mentions_embeddings_tensor
    gc.collect()

    # Step 5: Add predictions and confidence scores to mention pairs DataFrame
    predicted_mention_pairs = mention_pairs_df.copy()
    predicted_mention_pairs['model_prediction'] = predictions
    predicted_mention_pairs['confidence'] = abs(predicted_mention_pairs['model_prediction'].copy() - 0.5) * 2
    predicted_mention_pairs['coreference_prediction'] = (predicted_mention_pairs['model_prediction'] >= 0.5) * 1

    predicted_mention_pairs = rank_predicted_antecedents(predicted_mention_pairs)

    columns = ['text', 'gender', 'number', 'prop']
    # Extract information for mentions in pairs
    for mention_polarity in ["A", "B"]:
        predicted_mention_pairs = predicted_mention_pairs.merge(
            CAT_entities_df[columns].add_prefix(f"{mention_polarity}_"),
            left_on=mention_polarity,
            right_index=True,
            how='left')

    print("Postprocessing Mention Pairs Predictions...")
    # Step 6: Post-process the mention pairs to refine predictions
    predicted_mention_pairs = postprocess_mentions_pairs(predicted_mention_pairs,
                                                         CAT_entities_df=CAT_entities_df,
                                                         propagate_coref=propagate_coref,
                                                         rule_based_postprocess=rule_based_postprocess)

    if not characters_alias_list is None:
        predicted_mention_pairs = use_characters_alias_list(characters_alias_list, CAT_entities_df, predicted_mention_pairs)

    # Step 7: Generate the coreference matrix from processed mention pairs
    predicted_coreference_matrix = generate_coreference_matrix_with_cache(CAT_entities_df,
                                                                          predicted_mention_pairs,
                                                                          verbose=verbose,
                                                                          confidence_threshold=0)

    # Step 8: Generate clusters from the coreference matrix
    clusters = generate_clusters(predicted_coreference_matrix)

    # Step 9: Create a mapping dictionary for CAT_entities_df indices to mention IDs
    CAT_entities_df_ids = list(entities_df[entities_df["cat"].isin(entity_types)].copy().index)
    COREF_mapping_dict = {CAT_id: entities_id for CAT_id, entities_id in enumerate(CAT_entities_df_ids)}

    # Step 10: Assign cluster IDs to the original entities DataFrame
    entities_df["COREF"] = None  # Initialize COREF column

    for cluster_id, cluster in enumerate(clusters):
        # Map cluster mention IDs back to the original entity indices
        entities_ids = [COREF_mapping_dict[mention_id] for mention_id in cluster]
        # Assign the cluster ID to all entities in the current cluster
        entities_df.loc[entities_ids, "COREF"] = cluster_id

    # Return the updated entities DataFrame
    return entities_df