import os
from tqdm.auto import tqdm
import pickle
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import rnn
import torch.nn as nn
import torch.nn.functional as F
from TorchCRF import CRF
import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tabulate import tabulate
import gc
from transformers import AutoConfig

from .propp_fr_load_save_functions import load_tokens_df, load_entities_df, load_text_file
from .propp_fr_generate_tokens_embeddings_tensor import load_tokenizer_and_embedding_model, get_embedding_tensor_from_tokens_df


#%%
def get_NER_training_dictionary(files_directory,
                                NER_training_dictionary_path,
                                model_name,
                                subword_pooling_strategy="average",
                                verbose=1):
    """
    Generate, save, and load a dictionary for NER training containing token embeddings, token data, and entity data.

    This function creates a dictionary where each file in the directory is processed to generate token embeddings and
    other data required for Named Entity Recognition (NER) model training. The dictionary is saved to disk for reuse,
    avoiding redundant computation.

    Parameters:
    ----------
    files_directory : str
        Path to the directory containing token and entity files. Token files should have a `.tokens` extension.

    NER_training_dictionary_path : str
        Path to save/load the serialized dictionary (Pickle format) containing the processed NER data.

    tokenizer : Tokenizer
        A tokenizer compatible with the specified `model`, used for processing tokens.

    model : nn.Module
        A model for generating token embeddings (e.g., a transformer model like BERT or RoBERTa).

    Returns:
    -------
    dict
        A dictionary where keys are file names (without extensions) and values are dictionaries containing:
        - "tokens_df": A DataFrame with tokenized data for each file.
        - "entities_df": A DataFrame with entity data for each file.
        - "tokens_embeddings_tensor": A tensor with token embeddings for each file.
    """

    # Define the expected file extension for token files
    extension = ".tokens"

    # Extract all .token file names (without the extension) from the directory
    tokens_files = sorted([f.replace(extension, "") for f in os.listdir(files_directory) if f.endswith(extension)])

    # If the dictionary doesn't exist, create it
    if not os.path.exists(NER_training_dictionary_path):
        NER_training_dictionary = {}
        print(f"Loading {model_name} to initialize training dictionary")
        tokenizer, model = load_tokenizer_and_embedding_model(model_name=model_name)

        for file_name in tqdm(tokens_files, desc="Generating Tokens Embeddings"):
            text = load_text_file(file_name, files_directory)
            tokens_df = load_tokens_df(file_name, files_directory)
            entities_df = load_entities_df(file_name, files_directory)

            tokens_embeddings_tensor = get_embedding_tensor_from_tokens_df(text,
                                                               tokens_df,
                                                               tokenizer,
                                                               model,
                                                               sliding_window_size='max',
                                                               mini_batch_size=12,
                                                               sliding_window_overlap=0.5,
                                                               subword_pooling_strategy=subword_pooling_strategy, # ["average", "first", "last", "first_last"]
                                                               device=None,
                                                               verbose=verbose)

            NER_training_dictionary[file_name] = {"tokens_df": tokens_df,
                                                  "entities_df": entities_df,
                                                  "tokens_embeddings_tensor": tokens_embeddings_tensor}

        with open(NER_training_dictionary_path, "wb") as file:
            pickle.dump(NER_training_dictionary, file)

        # 1. Delete the model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    else:
        with open(NER_training_dictionary_path, "rb") as file:
            NER_training_dictionary = pickle.load(file)

    return NER_training_dictionary

#%%
def get_BIOES_tags_list(entities_df, tokens_df, nested_level=0, NER_cat_list=None, tagging_scheme="BIOES"):
    """
    Convert entity indices into a list of BIO, BIOE, or BIOES tags for NER model training.

    This function takes DataFrames for entities and tokens, processes entity boundaries,
    and converts them into a sequence of tags following a specified tagging scheme. The
    resulting tag list is used for training Named Entity Recognition (NER) models.

    Parameters:
    ----------
    entities_df : pd.DataFrame
        A DataFrame containing entity information with columns:
        - 'start_token': The starting token index of the entity.
        - 'end_token': The ending token index of the entity.
        - 'mention_len': Length of the entity (in tokens).
        - 'cat': The category of the entity (e.g., 'PER', 'LOC').
        - 'in_to_out_nested_level': Level of nesting for hierarchical entities.

    tokens_df : pd.DataFrame
        A DataFrame containing token information with token indices that will be tagged.

    nested_level : int, optional, default=0
        Specify the nested level to filter entities (e.g., inner entities = 0).

    NER_cat_list : list, optional, default=None
        A list of valid entity categories to consider (e.g., ['PER', 'LOC', 'GPE']).
        If None, no category filtering is applied.

    tagging_scheme : str, optional, default="BIOES"
        The tagging scheme to use:
        - "BIO": Beginning, Inside, Outside.
        - "BIOE": Beginning, Inside, End.
        - "BIOES": Beginning, Inside, End, Single-token entity.

    Returns:
    -------
    list
        A list of tags (one for each token in `tokens_df`) following the specified tagging scheme.
    """
    if NER_cat_list is None:
        NER_cat_list = ['PER', 'LOC', 'FAC', 'TIME', 'VEH', 'GPE']  # Default categories

    # Filter entities by nesting level and categories
    nested_level_entities_df = entities_df[(entities_df['in_to_out_nested_level'] == nested_level)
                                           & (entities_df['cat'].isin(NER_cat_list))].copy()
    # Initialize BIOES tags and entity categories
    tokens_df["BIOES"] = "O"  # Default tag: "Outside"
    tokens_df["cat"] = ""  # Default category: None

    # Dictionaries to store token indices for each tag/category
    tag_dict = {"B": [], "I": [], "S": [], "E": []}
    cat_dict = {label: [] for label in NER_cat_list}

    if tagging_scheme == "BIOES":
        for start_token, end_token, mention_len, cat in nested_level_entities_df[
            ['start_token', 'end_token', 'mention_len', 'cat']].values:
            tokens_ids = list(range(start_token, end_token + 1))
            cat_dict[cat] += tokens_ids
            if mention_len == 1:
                tag_dict["S"] += tokens_ids
            else:
                tag_dict["B"] += [start_token]
                tag_dict["E"] += [end_token]
                if mention_len > 2:
                    tag_dict["I"] += tokens_ids[1:-1]

    elif tagging_scheme == "BIOE":
        for start_token, end_token, mention_len, cat in nested_level_entities_df[
            ['start_token', 'end_token', 'mention_len', 'cat']].values:
            tokens_ids = list(range(start_token, end_token + 1))
            cat_dict[cat] += tokens_ids
            tag_dict["B"] += [start_token]
            tag_dict["E"] += [end_token]
            if mention_len > 2:
                tag_dict["I"] += tokens_ids[1:-1]

    elif tagging_scheme == "BIO":
        for start_token, end_token, mention_len, cat in nested_level_entities_df[
            ['start_token', 'end_token', 'mention_len', 'cat']].values:
            tokens_ids = list(range(start_token, end_token + 1))
            cat_dict[cat] += tokens_ids
            tag_dict["B"] += [start_token]
            if mention_len > 1:
                tag_dict["I"] += tokens_ids[1:]

    for tag in tag_dict.keys():
        tokens_df.loc[tag_dict[tag], 'BIOES'] = tag
    for cat in cat_dict.keys():
        tokens_df.loc[cat_dict[cat], 'cat'] = f"-{cat}"

    # Combine tags with category labels (e.g., "B-PER")
    tokens_df["BIOES"] = tokens_df["BIOES"] + tokens_df["cat"]

    BIOES_tag_list = tokens_df["BIOES"].tolist()

    return BIOES_tag_list

#%%
def prepare_NER_dataset(nested_level=0,
                        NER_cat_list=['PER', 'LOC', 'FAC', 'TIME', 'VEH', 'GPE'],
                        tagging_scheme="BIOES",
                        NER_training_dictionary=None):
    """
    Prepares the NER dataset for training by generating BIOES tag mappings and aligning embeddings with labels.

    Args:
        nested_level (int): The level of nested entities to consider. Default is 0 (non-nested entities).
        NER_cat_list (list): List of entity categories to include in the tagging process.
        tagging_scheme (str): Tagging scheme to use for labeling ('BIO', 'BIOE', 'BIOES').
        NER_training_dictionary (dict): Dictionary containing tokens, entities, and embeddings for each file.

    Returns:
        tuple: A tuple containing:
            - Updated NER_training_dictionary (dict): Includes the BIOES tags and prepared datasets.
            - id2label (dict): Mapping of label IDs to BIOES tags.
            - label2id (dict): Mapping of BIOES tags to label IDs.
    """
    if NER_training_dictionary is None:
        raise ValueError("NER_training_dictionary cannot be None. Provide a valid dictionary.")

    all_labels = []

    # Step 1: Generate BIOES tags for each file in the dictionary
    for file_name in NER_training_dictionary.keys():
        tokens_df, entities_df = NER_training_dictionary[file_name]["tokens_df"], NER_training_dictionary[file_name][
            "entities_df"]
        # Generate BIOES tags using helper function
        BIOES_tag_list = get_BIOES_tags_list(entities_df,
                                             tokens_df,
                                             nested_level=nested_level,
                                             NER_cat_list=NER_cat_list,
                                             tagging_scheme=tagging_scheme)
        # Store BIOES tags in the dictionary
        NER_training_dictionary[file_name]["BIOES_tag_list"] = BIOES_tag_list
        all_labels += BIOES_tag_list

    # Step 2: Create mappings between labels and IDs
    all_labels = dict(Counter(all_labels).most_common())  # Count occurrences of each label
    id2label = {i: label for i, label in enumerate(all_labels.keys())}
    label2id = {label: i for i, label in id2label.items()}

    # Step 3: Align embeddings with BIOES tags for dataset preparation
    for file_name in NER_training_dictionary.keys():
        tokens_df, tokens_embeddings_tensor, BIOES_tag_list = NER_training_dictionary[file_name]["tokens_df"], \
        NER_training_dictionary[file_name]["tokens_embeddings_tensor"], NER_training_dictionary[file_name][
            "BIOES_tag_list"]

        # Convert BIOES tags to IDs
        BIOES_tag_ids = np.array([label2id[tag] for tag in BIOES_tag_list])
        BIOES_tag_ids_tensor = torch.tensor(BIOES_tag_ids, dtype=torch.long)

        # Prepare the dataset by grouping tokens by sentence
        dataset = []
        for sentence_ID, sentence_tokens_df in tokens_df.groupby("sentence_ID"):
            tokens_ids = sentence_tokens_df['token_ID_within_document'].tolist()
            dataset.append({"embeddings": tokens_embeddings_tensor[tokens_ids],
                            "BIOES_tags": BIOES_tag_ids_tensor[tokens_ids]})

        # Store the dataset in the dictionary
        NER_training_dictionary[file_name]["dataset"] = dataset

    return NER_training_dictionary, id2label, label2id

#%%
class NERDataset(Dataset):
    """
    Custom Dataset for Named Entity Recognition (NER) tasks.

    Args:
        dataset (list): List of samples, where each sample contains embeddings and BIOES tags.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves the embeddings and BIOES tags for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (embeddings, BIOES_tags)
        """
        item = self.dataset[idx]
        return item["embeddings"], item["BIOES_tags"]
def collate_fn(batch):
    """
    Custom collate function for batching NER data.

    Pads sequences of embeddings, labels, and generates an attention mask.

    Args:
        batch (list): List of tuples (embeddings, BIOES_tags).

    Returns:
        tuple:
            - embeddings_padded (torch.Tensor): Padded embeddings.
            - labels_padded (torch.Tensor): Padded BIOES tags.
            - attention_mask_padded (torch.Tensor): Padded attention mask.
    """
    # Separate embeddings and labels from the batch
    embeddings = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Sequence lengths for attention mask
    seq_lengths = [len(embedding) for embedding in embeddings]

    # Generate attention masks
    attention_mask = [torch.ones(length) for length in seq_lengths]

    # Pad embeddings and labels
    embeddings_padded = pad_sequence(embeddings, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

    # Pad attention masks
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return embeddings_padded, labels_padded, attention_mask_padded
def prepare_data_loader(all_files,
                        test_files,
                        NER_training_dictionary=None,
                        batch_size=16,
                        train_with_validation=0.85):
    """
    Prepares DataLoaders for training and validation.

    Args:
        all_files (list): List of all file names in the dataset.
        test_files (list): List of file names reserved for testing.
        NER_training_dictionary (dict): Dictionary containing datasets for each file.
        batch_size (int): Number of samples per batch. Default is 8.
        train_with_validation (float): Proportion of training data used for training (remainder for validation). Default is 0.85.

    Returns:
        tuple:
            - train_loader (DataLoader): DataLoader for training.
            - validation_loader (DataLoader): DataLoader for validation.
    """
    if NER_training_dictionary is None:
        raise ValueError("NER_training_dictionary cannot be None. Provide a valid dictionary.")

    # Separate files into training and validation sets
    train_files = [file_name for file_name in all_files if not file_name in test_files]

    # Aggregate all samples from training files
    all_samples = [sample
                   for file_name in train_files
                   for sample in NER_training_dictionary[file_name]["dataset"]
                   ]

    # Shuffle the training samples
    random.shuffle(all_samples)

    # Split into training and validation datasets
    split_index = int(len(all_samples) * train_with_validation)
    train_data = all_samples[:split_index]
    validation_data = all_samples[split_index:]

    # Create Dataset objects
    train_dataset = NERDataset(train_data)
    validation_dataset = NERDataset(validation_data)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, validation_loader

#%%
# Initialize the model class
class LockedDropout(nn.Module):
    """
    Implements Locked Dropout for regularization in sequential models.
    The dropout mask is consistent across time steps to preserve sequence structure.

    Args:
        p (float): Dropout probability (between 0 and 1).
    """

    def __init__(self, p):
        super(LockedDropout, self).__init__()
        self.p = p

    def forward(self, x):
        """
        Applies the Locked Dropout to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: Tensor after applying Locked Dropout.
        """
        if not self.training or self.p == 0:
            return x  # No dropout during evaluation or if p = 0
        # Create a dropout mask that is the same across all time steps
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)  # Scale mask to preserve magnitude
        return x * mask
class Highway(nn.Module):
    """
    Implements a Highway Network layer, allowing a blend of transformation and preservation of input features.

    Args:
        input_size (int): Size of the input features.
        output_size (int or None): Size of the output features. Defaults to input_size.
        activation (Callable): Activation function for the transform layer. Defaults to ReLU.
    """

    def __init__(self,
                 input_size,
                 output_size=None,
                 activation=F.relu):
        super(Highway, self).__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.activation = activation

        # Linear layer for optional dimension projection
        if self.input_size != self.output_size:
            self.project = nn.Linear(self.input_size, self.output_size)
        else:
            self.project = None

        # Layers for the transform and gate
        self.transform = nn.Linear(self.output_size, self.output_size)
        self.gate = nn.Linear(self.output_size, self.output_size)

    def forward(self, x):
        """
        Applies the Highway transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, seq_len, output_size).
        """
        # Optional projection
        if self.project is not None:
            x = self.project(x)  # Optional projection to output_size

        # Transform and gate outputs
        transform_out = self.activation(self.transform(x))
        gate_out = torch.sigmoid(self.gate(x))

        # Highway transformation
        return gate_out * transform_out + (1 - gate_out) * x
class NERModel(nn.Module):
    """
    Named Entity Recognition model with BiLSTM and CRF layers for sequence labeling tasks.

    Args:
        embedding_dim (int): Dimension of input embeddings.
        num_labels (int): Number of output labels for classification.
        locked_dropout (float): Dropout probability for LockedDropout layers.
        embedding2nn_type (str): Type of embedding transformation ('highway' or 'linear').
        embedding2nn_dim (int): Dimension of intermediate feature representation.
        LSTM_output (int): Number of LSTM hidden units (per direction).
        LSTM_layers (int): Number of stacked LSTM layers.
    """

    def __init__(self,
                 embedding_dim,
                 num_labels,
                 locked_dropout=0.5,
                 embedding2nn_type="highway",
                 embedding2nn_dim=2048,
                 LSTM_output=256,
                 LSTM_layers=1
                 ):
        super(NERModel, self).__init__()
        self.LSTM_output = LSTM_output
        self.num_labels = num_labels

        # Locked Dropout Layer for Regularization
        self.locked_dropout = LockedDropout(p=locked_dropout)

        # Feature transformation layer
        if embedding2nn_type == "highway":
            self.embedding2nn = Highway(embedding_dim, embedding2nn_dim)  # Highway layer
        elif embedding2nn_type == "linear":
            self.embedding2nn = nn.Linear(embedding_dim, embedding2nn_dim)
        else:
            raise ValueError(f"Unsupported embedding2nn_type: {embedding2nn_type}")

        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size=embedding2nn_dim,  # Input dimension (e.g., from embeddings)
                            hidden_size=LSTM_output,
                            # Hidden size for each direction -> the output dimension is hidden size * 2
                            num_layers=LSTM_layers,  # Number of LSTM layers
                            bidirectional=True,  # Make it bidirectional
                            batch_first=True)  # Input tensor format: (batch, seq_len, feature_dim)

        # Linear Layer to Project BiLSTM Outputs to Label Space
        self.linear = nn.Linear(LSTM_output * 2, num_labels)

        # CRF Layer for Sequence Labeling
        self.crf = CRF(num_labels)

    def forward(self,
                embeddings,
                attention_mask=None,
                labels=None
                ):
        """
        Forward pass of the NER model.

        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, seq_len, embedding_dim).
            attention_mask (torch.Tensor, optional): Mask for valid tokens, shape (batch_size, seq_len).
            labels (torch.Tensor, optional): Ground-truth labels, shape (batch_size, seq_len).

        Returns:
            torch.Tensor or tuple:
                - Training mode: Returns CRF loss (torch.Tensor).
                - Evaluation mode: Returns predicted labels (list) and confidences (list).
        """
        # Apply LockedDropout to embeddings
        embeddings = self.locked_dropout(embeddings)  # Shape: (batch_size, seq_len, embedding_dim)
        # Project embeddings to intermediate representation
        embeddings = self.embedding2nn(embeddings)  # Shape: (batch_size, seq_len, 1536)

        # BiLSTM for sequential context
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embeddings)  # Shape: (batch_size, seq_len, hidden_size * 2)

        # Apply LockedDropout to LSTM output
        lstm_out = self.locked_dropout(lstm_out)

        # Project LSTM outputs to label space
        logits = self.linear(lstm_out)

        if labels is not None:
            # Compute CRF loss during training
            mask = attention_mask.bool()  # Convert attention mask to boolean
            loss = -self.crf(logits, labels, mask=mask) / lstm_out.size(0)  # CRF loss normalized by batch size
            return loss
        else:
            # Decode predictions and compute confidences during inference
            mask = attention_mask.bool()
            predicted_labels = self.crf.viterbi_decode(logits, mask)

            # Apply softmax to get the probabilities for each label
            softmax_probs = torch.softmax(logits, dim=-1)

            confidences = []
            for batch_idx in range(softmax_probs.size(0)):  # Iterate over the batch
                sentence_predicted_labels = predicted_labels[batch_idx]  # Labels for one sentence
                sentence_softmax_probs = softmax_probs[batch_idx]  # Probabilities for one sentence

                sentence_confidences = []  # Store confidences for this sentence
                for token_idx, label in enumerate(sentence_predicted_labels):  # Iterate over tokens
                    confidence = sentence_softmax_probs[token_idx, label].item()  # Extract confidence
                    sentence_confidences.append(confidence)
                confidences.append(sentence_confidences)  # Append confidences for this sentence

            return predicted_labels, confidences

#%%
def remove_BIO_illegal_transitions(BIO_tag_list, tagging_scheme="BIOES"):
    """
    Cleans a list of BIO tags by removing illegal transitions according to the specified tagging scheme.

    Parameters:
        bio_tag_list: list
            A list of predicted BIO tags (e.g., ["B-PER", "I-PER", "O", "E-PER", ...]).
        tagging_scheme: str, optional
            The tagging scheme used for the labels. Defaults to "BIOES".

    Returns:
        list:
            A list of cleaned BIO tags where illegal transitions are corrected to "O".
    """
    if tagging_scheme not in ["BIO", "BIOE", "BIOES"]:
        raise ValueError(f"Unsupported tagging scheme: {tagging_scheme}. Must be one of ['BIO', 'BIOE', 'BIOES'].")

    cleaned_tags_list = []  # To store corrected tags
    previous_tag, previous_cat = None, None  # Track the previous tag and category

    for BIO_tag in BIO_tag_list:
        if BIO_tag == "O":  # Handle the "O" tag (no entity)
            cleaned_tags_list.append(BIO_tag)
            previous_tag, previous_cat = None, None  # Reset tracking for next entity
            continue

        else:
            tag, cat = BIO_tag.split("-")
            if tag in ["B", "S"]:  # Handle valid starting tags ("B", "S")
                cleaned_tags_list.append(BIO_tag)
                previous_tag, previous_cat = tag, cat
                continue

            elif tag in ["I", "E"]:  # Handle continuation tags ("I", "E")
                if previous_tag in ["B",
                                    "I"] and previous_cat == cat:  # Valid continuation if preceded by "B" or "I" of the same category
                    cleaned_tags_list.append(BIO_tag)
                    previous_tag, previous_cat = tag, cat
                else:  # Illegal continuation, reset to "O"
                    cleaned_tags_list.append("O")
                    previous_tag, previous_cat = None, None

    return cleaned_tags_list

#%%
def extract_entities_from_BIO_tag_list(BIO_tag_list):
    """
    Extracts entity boundaries and categories from a list of BIOES tags.

    Parameters:
        BIO_tag_list (List[str]):
            A list of BIOES tags (e.g., ["B-PER", "I-PER", "E-PER", "S-LOC", "O"]).

    Returns:
        pd.DataFrame:
            A DataFrame with columns ["start_token", "end_token", "cat"] where:
            - "start_token": The starting index of the entity.
            - "end_token": The ending index of the entity.
            - "cat": The category of the entity.
    """
    # List to store identified entities
    entities = []

    # Variables to track the current entity's state
    previous_tag, previous_cat = None, None
    start_token = None
    open_entity = False  # Tracks whether we are inside an entity

    for i, BIO_tag in enumerate(BIO_tag_list):
        if open_entity == False:
            if BIO_tag == "O":  # "O" means outside of any entity
                previous_tag, previous_cat = None, None
                continue
            else:
                tag, cat = BIO_tag.split("-")
                if tag == "S":  # "S-X" represents a single-token entity
                    entities.append({"start_token": i,
                                     "end_token": i,
                                     "cat": cat})
                    previous_tag, previous_cat = None, None
                    continue
                elif tag == "B":  # "B-X" starts a new entity
                    open_entity = True
                    start_token = i
                    previous_cat = cat
                    continue

        else:  # An entity is currently open
            if BIO_tag == "O":  # End the current entity if an "O" is encountered
                entities.append({"start_token": start_token,
                                 "end_token": i - 1,
                                 "cat": previous_cat})
                open_entity = False
                previous_tag, previous_cat = None, None
                continue
            else:
                tag, cat = BIO_tag.split("-")
                if tag == "S":  # Close the previous entity and add the current single-token entity
                    entities.append({"start_token": start_token,
                                     "end_token": i - 1,
                                     "cat": previous_cat})
                    open_entity = False

                    entities.append({"start_token": i,
                                     "end_token": i,
                                     "cat": cat})
                    previous_tag, previous_cat = None, None
                    continue
                elif tag == "B":  # Close the previous entity and start a new one
                    entities.append({"start_token": start_token,
                                     "end_token": i - 1,
                                     "cat": previous_cat})
                    open_entity = True
                    start_token = i
                    previous_cat = cat
                    continue

                elif tag == "I":  # Continuation of the current entity
                    open_entity = True
                    previous_cat = cat
                    continue

                elif tag == "E":  # Close the current entity
                    open_entity = False
                    entities.append({"start_token": start_token,
                                     "end_token": i,
                                     "cat": cat})
                    previous_tag, previous_cat = None, None
                    continue

    # Handle case where the last entity might not have been closed
    if open_entity == True:
        entities.append({"start_token": start_token,
                         "end_token": i,
                         "cat": previous_cat})

    entities_df = pd.DataFrame(entities, columns=["start_token", "end_token", "cat"])

    return entities_df

#%%
def combine_gold_and_predicted_entities_df(gold_entities_df, predicted_entities_df):
    """
    Combines two DataFrames containing gold-standard and predicted entities,
    aligning them based on 'start_token' and 'end_token'. Fills missing values
    for categorical columns and ensures confidence values are present.

    Parameters:
    - gold_entities_df: DataFrame
        Contains the gold-standard entity annotations with columns like 'start_token', 'end_token', 'cat', etc.
    - predicted_entities_df: DataFrame
        Contains the predicted entity annotations with columns like 'start_token', 'end_token', 'cat', 'confidence', etc.

    Returns:
    - entities_df: DataFrame
        Merged DataFrame containing aligned gold and predicted entities with
        missing values filled for 'cat' and 'confidence' columns.
    """
    # Merge the two DataFrames on 'start_token' and 'end_token'
    entities_df = pd.merge(
        gold_entities_df,
        predicted_entities_df,
        on=['start_token', 'end_token'],
        how='outer',
        suffixes=('_gold', '_predicted')
    )

    # Ensure all relevant columns are present and handle missing data
    for column in ['cat_gold', 'cat_predicted']:
        if column not in entities_df.columns:
            entities_df[column] = 'O'  # Add missing column with default value
        else:
            entities_df[column] = entities_df[column].fillna('O')  # Fill missing values in existing column

        # Explicitly convert 'confidence' to a numeric type to avoid deprecation warning
    if 'confidence' in entities_df.columns:
        entities_df['confidence'] = pd.to_numeric(entities_df['confidence'], errors='coerce').fillna(0)
    else:
        entities_df['confidence'] = 0  # Add confidence column with default value

    # Optional: Reorder columns for clarity (if needed)
    column_order = ['start_token', 'end_token', 'cat_gold', 'cat_predicted', 'confidence']
    entities_df = entities_df[[col for col in column_order if col in entities_df.columns]]

    return entities_df

#%%
def get_NER_metrics_df(entities_df, NER_cat_list=None):
    def get_classification_metrics(entities_df, label):
        # Calculate the true positives, false positives, true negatives, and false negatives
        TP = len(entities_df[(entities_df['cat_gold'] == label) & (entities_df['cat_predicted'] == label)])
        FP = len(entities_df[(entities_df['cat_gold'] != label) & (entities_df['cat_predicted'] == label)])
        TN = len(entities_df[(entities_df['cat_gold'] != label) & (entities_df['cat_predicted'] != label)])
        FN = len(entities_df[(entities_df['cat_gold'] == label) & (entities_df['cat_predicted'] != label)])

        # Calculate support, accuracy, precision, recall, and F1 score
        support = len(entities_df[entities_df['cat_gold'] == label])
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return support, accuracy, precision, recall, f1_score

    if NER_cat_list == None:
        NER_cat_list = set(entities_df["cat_gold"]) | set(entities_df["cat_predicted"])
        NER_cat_list = [cat for cat in NER_cat_list if cat not in ["O"]]
    # Initialize a list to store metrics for each NER category
    metrics = []
    for ner_tag in NER_cat_list:
        support, accuracy, precision, recall, f1_score = get_classification_metrics(entities_df, ner_tag)
        metrics.append({
            'NER_tag': ner_tag,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': support
        })

    # Create a DataFrame to store the metrics for each category
    metrics_df = pd.DataFrame(metrics).sort_values('support', ascending=False)
    metrics_df.index = metrics_df['NER_tag']
    metrics_df = metrics_df.drop("NER_tag", axis=1)

    # # Calculate micro averages
    # for metric in ['precision', 'recall', 'f1_score']:
    #     metrics_df.loc["micro_avg", metric] = (metrics_df[metric] * metrics_df["support"]).sum() / metrics_df[
    #         "support"].sum()
    # metrics_df.loc["micro_avg", "support"] = metrics_df["support"].sum()
    #
    # # Filter out categories with zero support for macro averages
    # non_empty_support_tags = [ner_tag for ner_tag in NER_cat_list if metrics_df.loc[ner_tag, "support"] > 0]
    #
    # # Calculate macro averages
    # for metric in ['precision', 'recall', 'f1_score']:
    #     metrics_df.loc["macro_avg", metric] = metrics_df.loc[non_empty_support_tags, metric].mean()
    # metrics_df.loc["macro_avg", "support"] = metrics_df.loc[non_empty_support_tags, "support"].sum()
    #
    # return metrics_df
    # Fill metrics with 1 where support == 0
    for metric in ['precision', 'recall', 'f1_score']:
        metrics_df.loc[metrics_df["support"] == 0, metric] = 1.0

    total_support = metrics_df["support"].sum()

    for metric in ['precision', 'recall', 'f1_score']:
        if total_support > 0:
            metrics_df.loc["micro_avg", metric] = (metrics_df[metric] * metrics_df["support"]).sum() / total_support
        else:
            metrics_df.loc["micro_avg", metric] = 1.0  # or 0.0 or np.nan depending on your convention

    metrics_df.loc["micro_avg", "support"] = total_support

    # Filter out only rows with non-zero support for macro average
    non_empty_support_tags = [ner_tag for ner_tag in NER_cat_list if metrics_df.loc[ner_tag, "support"] > 0]

    # Handle the case where no class has support > 0
    if non_empty_support_tags:
        for metric in ['precision', 'recall', 'f1_score']:
            metrics_df.loc["macro_avg", metric] = metrics_df.loc[non_empty_support_tags, metric].mean()
        metrics_df.loc["macro_avg", "support"] = metrics_df.loc[non_empty_support_tags, "support"].sum()
    else:
        for metric in ['precision', 'recall', 'f1_score']:
            metrics_df.loc["macro_avg", metric] = 0.0
        metrics_df.loc["macro_avg", "support"] = 0

    return metrics_df


#%%
def compute_NER_validation_metrics(all_predictions, all_labels, id2label,
                                   NER_cat_list=['PER', 'LOC', 'FAC', 'TIME', 'VEH', 'GPE']):
    # Convert all IDs to BIO tags using id2label
    def convert_to_BIO_tags(sequences):
        return [[id2label[id] for id in sequence] for sequence in sequences]

    predicted_BIO_tags = convert_to_BIO_tags(all_predictions)
    gold_BIO_tags = convert_to_BIO_tags(all_labels)

    # Remove illegal transitions in BIO tags for predictions
    legal_predicted_BIO_tags = [remove_BIO_illegal_transitions(sentence) for sentence in predicted_BIO_tags]

    # Flatten gold and predicted BIO tags for entity extraction
    flatten_gold_BIO_tags = [label for sentence in gold_BIO_tags for label in sentence]
    flatten_legal_predicted_BIO_tags_predictions = [label for sentence in legal_predicted_BIO_tags for label in
                                                    sentence]

    # Extract entities from BIO tags and combine them
    gold_entities_df = extract_entities_from_BIO_tag_list(flatten_gold_BIO_tags)
    predicted_entities_df = extract_entities_from_BIO_tag_list(flatten_legal_predicted_BIO_tags_predictions)
    predicted_entities_df["confidence"] = None
    inferred_entities_df = combine_gold_and_predicted_entities_df(gold_entities_df, predicted_entities_df)

    # Get the NER metrics (precision, recall, F1 scores per entity category)
    metrics_df = get_NER_metrics_df(inferred_entities_df, NER_cat_list=NER_cat_list)

    # Extract entity-based F1 scores
    entities_micro_f1 = metrics_df.loc['micro_avg', 'f1_score']
    entities_macro_f1 = metrics_df.loc['macro_avg', 'f1_score']

    # Filter out non-entity pairs for precision, recall, and F1
    filtered_predictions = []
    filtered_labels = []
    for pred, label in zip(flatten_legal_predicted_BIO_tags_predictions, flatten_gold_BIO_tags):
        if not (pred == 0 and label == 0):  # Skip non-entity pairs
            filtered_predictions.append(pred)
            filtered_labels.append(label)

    # Compute precision, recall, and F1 for each class
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(filtered_labels, filtered_predictions,
                                                                                 average='micro', zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(filtered_labels, filtered_predictions,
                                                                                 average='macro', zero_division=0)

    return micro_f1, macro_f1, entities_micro_f1, entities_macro_f1

#%%
def get_predicted_entities_df(NER_model, tokens_df, sentences_embeddings, id2label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the model to evaluation mode
    NER_model.eval()

    all_predictions = []
    all_confidences = []

    # Iterate over unique sentence IDs in tokens_df
    for sentence_id in tokens_df['sentence_ID'].unique():
        # Get the embeddings for the current sentence from sentences_embeddings
        sentence_embeddings = sentences_embeddings[sentence_id]
        sentence_embeddings = sentence_embeddings.to(dtype=torch.float32)
        sentence_attention_mask = torch.ones(sentence_embeddings.shape[0])  # Create an attention mask of ones

        # Add batch dimension (batch_size=1)
        sentence_embeddings = sentence_embeddings.unsqueeze(0)  # Shape becomes (1, seq_len, embedding_dim)
        sentence_attention_mask = sentence_attention_mask.unsqueeze(0)  # Shape becomes (1, seq_len)

        # Move tensors to the correct device (GPU or CPU)

        sentence_embeddings = sentence_embeddings.to(device)
        sentence_attention_mask = sentence_attention_mask.to(device)

        # Perform inference
        with torch.no_grad():  # No need to track gradients during inference
            predictions, confidences = NER_model(sentence_embeddings, attention_mask=sentence_attention_mask)

        # Append the predictions and confidences to the respective lists
        all_predictions += predictions
        all_confidences += confidences

    # Convert predictions to BIO tags using id2label mapping
    predicted_BIO_tags = [
        [id2label[id] for id in sequence]
        for sequence in all_predictions
    ]

    # Remove illegal transitions in BIO tags
    legal_predicted_BIO_tags = [remove_BIO_illegal_transitions(sentence) for sentence in predicted_BIO_tags]

    # Flatten the BIO tag sequences for entity extraction
    flatten_legal_predicted_BIO_tags_predictions = [label for sentence in legal_predicted_BIO_tags for label in
                                                    sentence]

    # Extract entities from the flattened BIO tag list
    predicted_entities_df = extract_entities_from_BIO_tag_list(flatten_legal_predicted_BIO_tags_predictions)

    # Flatten the confidences list
    confidences = [confidence for sequence in all_confidences for confidence in sequence]

    # Calculate the confidence for each predicted entity
    mentions_confidence = []
    for start_token, end_token in predicted_entities_df[['start_token', 'end_token']].values:
        confidence_values = confidences[start_token:end_token + 1]
        mean_confidence = sum(confidence_values) / len(confidence_values)
        mentions_confidence.append(mean_confidence)

    # Add the confidence values to the predicted entities DataFrame
    predicted_entities_df['confidence'] = mentions_confidence

    return predicted_entities_df

def remove_overlapping_entities(all_predicted_entities):
    all_predicted_entities = all_predicted_entities.copy()  # Avoid modifying in place
    all_predicted_entities["overlapping_boundaries"] = False

    start_tokens = all_predicted_entities["start_token"].values
    end_tokens = all_predicted_entities["end_token"].values

    # Create a vectorized mask for overlapping entities
    n = len(all_predicted_entities)
    overlapping_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        start, end = start_tokens[i], end_tokens[i]

        # Vectorized comparison instead of DataFrame filtering
        higher_ranked_mask = np.arange(n) > i  # Entities ranked higher than i

        overlapping_mask |= (
            higher_ranked_mask
            & (
                ((start_tokens < start) & (end_tokens > start) & (end_tokens < end))
                | ((end_tokens > end) & (start_tokens > start) & (start_tokens < end))
                | ((start_tokens == start) & (end_tokens == end))
            )
        )

    # Assign overlapping flags in bulk
    all_predicted_entities.loc[overlapping_mask, "overlapping_boundaries"] = True

    # Filter and remove duplicates
    all_predicted_entities = all_predicted_entities[~all_predicted_entities["overlapping_boundaries"]].drop_duplicates()

    return all_predicted_entities


#%%
def combine_predicted_entities(predicted_entities_dfs, priority="confidence"):  # max_len, min_len
    # Combine all predicted entities into one DataFrame
    all_predicted_entities = pd.concat(predicted_entities_dfs, axis=0)

    # Calculate the mention length for each entity
    all_predicted_entities["mention_len"] = all_predicted_entities["end_token"] + 1 - all_predicted_entities[
        "start_token"]

    # Sort based on the priority
    if priority == "confidence":
        all_predicted_entities = all_predicted_entities.sort_values(['confidence'], ascending=[True]).reset_index(
            drop=True)
    if priority == "min_len":
        all_predicted_entities = all_predicted_entities.sort_values(['mention_len', 'confidence'],
                                                                    ascending=[False, True]).reset_index(drop=True)

    # # Initialize illegal boundary column to False
    # all_predicted_entities["overlapping_boundaries"] = False
    #
    # # Check for overlapping entities
    # for i, (start_token, end_token) in enumerate(
    #         tqdm(all_predicted_entities[['start_token', 'end_token']].values, desc="Removing Crossing Entities", leave=False)
    # ):
    #     # Your code here
    #
    #     # Find higher-ranked entities that overlap with the current entity
    #     higher_ranked_entities = all_predicted_entities.loc[i + 1:]
    #     overlapping_entities = higher_ranked_entities[
    #         ((higher_ranked_entities['start_token'] < start_token)
    #          & ((higher_ranked_entities['end_token'] > start_token)
    #             & (higher_ranked_entities['end_token'] < end_token)))
    #         | ((higher_ranked_entities['end_token'] > end_token)
    #            & ((higher_ranked_entities['start_token'] > start_token)
    #               & (higher_ranked_entities['start_token'] < end_token)))
    #         | ((higher_ranked_entities['start_token'] == start_token)
    #            & (higher_ranked_entities['end_token'] == end_token))
    #         ]
    #
    #     # Flag entities with overlapping boundaries
    #     if not overlapping_entities.empty:
    #         all_predicted_entities.loc[i, "overlapping_boundaries"] = True
    #
    # # Filter out entities with overlapping boundaries and remove duplicates
    # all_predicted_entities = all_predicted_entities[all_predicted_entities['overlapping_boundaries'] == False]
    # all_predicted_entities = all_predicted_entities.drop_duplicates()

    all_predicted_entities = remove_overlapping_entities(all_predicted_entities)

    # Retain only relevant columns and sort by token positions
    all_predicted_entities = all_predicted_entities[["start_token", "end_token", "cat", "confidence"]].sort_values(
        ["start_token", "end_token"], ascending=[True, True]).reset_index(drop=True)

    return all_predicted_entities

#%%
def train_NER_model(train_loader=None,
                    validation_loader=None,
                    embedding_dim=None,
                    num_labels=None,  # Number of unique BIOES labels
                    locked_dropout=0.5,
                    embedding2nn_type="highway",
                    embedding2nn_dim=2048,
                    LSTM_output=256,
                    LSTM_layers=1,
                    max_epoch=100,
                    learning_rate=0.00014,
                    loss_delta_patience=15,
                    validation_loss_patience=5,
                    metric="macro",
                    metric_patience=6,
                    NER_cat_list=['PER', 'LOC', 'FAC', 'TIME', 'VEH', 'GPE'],
                    id2label=None):
    # Choose device for training (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the NER model and move it to the chosen device
    model = NERModel(embedding_dim, num_labels, locked_dropout, embedding2nn_type, embedding2nn_dim, LSTM_output,
                     LSTM_layers).to(device)
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Initialize tracking variables
    precedent_validation_loss = float("+inf")
    no_improvement_counter = 0
    metric_patience_count = 0
    best_entities_f1 = float("-inf")
    best_model_state = None
    negative_loss_delta_count = 0
    logs = []  # List to store training logs

    # Training loop
    for epoch in tqdm(range(max_epoch)):
        print(f"EPOCH {epoch + 1}:", end="")

        model.train()  # Set model to training mode
        total_train_loss = 0

        # Loop over batches in the training data
        for embeddings, labels, attention_mask in tqdm(train_loader, desc="Training", leave=False):
            embeddings = embeddings.to(dtype=torch.float32)
            embeddings, labels, attention_mask = embeddings.to(device), labels.to(device), attention_mask.to(device)

            optimizer.zero_grad()  # Clear previous gradients

            # Forward pass
            train_loss = model(embeddings, attention_mask=attention_mask, labels=labels)

            # Ensure loss is scalar by averaging it over the batch
            train_loss = train_loss.mean()  # You can also use loss.sum() if preferred

            train_loss.backward()  # Backpropagation

            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # Update the model parameters

            total_train_loss += train_loss.item()  # Accumulate training loss

        # Calculate the average training loss
        train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        all_predictions, all_labels = [], []
        total_validation_loss = 0

        with torch.no_grad():  # Disable gradient calculation during validation to save memory
            for embeddings, labels, attention_mask in tqdm(validation_loader, desc="Evaluating", leave=False):
                embeddings = embeddings.to(dtype=torch.float32)
                embeddings, labels, attention_mask = embeddings.to(device), labels.to(device), attention_mask.to(device)

                # Forward pass for validation
                validation_loss = model(embeddings, attention_mask=attention_mask, labels=labels)
                validation_loss = validation_loss.mean()  # Ensure scalar loss

                total_validation_loss += validation_loss.item()

                # Get predictions (no labels passed during inference)
                predictions, _ = model(embeddings, attention_mask=attention_mask, labels=None)

                # Flatten and remove padding labels
                for i, prediction in enumerate(predictions):
                    valid_labels = labels[i][:len(prediction)]  # Slice to match prediction length
                    valid_labels = valid_labels[valid_labels != -1]  # Remove padding labels

                    all_predictions.append(prediction)
                    all_labels.append(valid_labels.cpu().numpy())

        # Calculate the average validation loss
        validation_loss = total_validation_loss / len(validation_loader)

        # Update learning rate based on validation loss
        scheduler.step(validation_loss)

        # Compute F1 scores for validation predictions
        micro_f1, macro_f1, entities_micro_f1, entities_macro_f1 = compute_NER_validation_metrics(all_predictions,
                                                                                                  all_labels, id2label,
                                                                                                  NER_cat_list=NER_cat_list)

        # Calculate the difference in loss between training and validation (loss delta) to track overfitting
        loss_delta = (train_loss - validation_loss)

        # Log metrics for the current epoch
        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "loss_delta": loss_delta,
            "entities_micro_f1": entities_micro_f1,
            "entities_macro_f1": entities_macro_f1,
            "negative_loss_delta_count": negative_loss_delta_count
        })

        # Select the F1 score based on the chosen metric (macro or micro)
        if metric == "macro":
            entities_f1 = entities_macro_f1
        elif metric == "micro":
            entities_f1 = entities_micro_f1

        # Update the best model state based on the F1 score
        if entities_f1 >= best_entities_f1:
            best_entities_f1 = entities_f1
            best_model_state = model.state_dict()  # Save the best model's state
            metric_patience_count = 0
        else:
            metric_patience_count += 1

        # Track the loss delta to detect potential training stagnation
        if loss_delta < 0:
            negative_loss_delta_count += 1
        else:
            negative_loss_delta_count = 0

        # Early stopping condition based on validation loss improvement
        if precedent_validation_loss < validation_loss:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0
            precedent_validation_loss = validation_loss

        # Print training and validation progress
        print(
            f"\tTrain Loss: {train_loss:.4f}\t\tVal Loss: {validation_loss:.4f}\t\tLoss Delta: {loss_delta:.4f}\t\tEntities micro_f1: {entities_micro_f1:.4f}\t\tEntities macro_f1: {entities_macro_f1:.4f}")

        # Check for early stopping criteria
        if negative_loss_delta_count == loss_delta_patience:
            print("loss_delta_patience reached - Stopping training.")
            break
        if metric_patience_count == metric_patience:
            print("metric_patience reached - Stopping training.")
            break
        if no_improvement_counter == validation_loss_patience:
            print("validation_loss_patience reached - Stopping training.")
            break

    # Restore the best model state after training
    model.load_state_dict(best_model_state)  # Load the best model's state
    model.to(device)  # Ensure the model is on the correct device (CPU/GPU)

    # Convert logs to a pandas DataFrame for easier analysis
    logs_df = pd.DataFrame(logs)

    return model, logs_df  # Return the trained model and the training logs

#%%
def mentions_detection_LOOCV_full_model_training(files_directory=None,
                                                 model_name="almanach/camembert-large",
                                                 trained_model_directory=None,
                                                 subword_pooling_strategy="first_last",
                                                 train_files="all",
                                                 test_files="all",
                                                 train_final_model = True,
                                                 train_with_validation=0.85,
                                                 NER_cat_list=None,
                                                 tagging_scheme="BIOES",
                                                 nested_levels=[0, 1],
                                                 batch_size=16,
                                                 learning_rate=0.00014,
                                                 locked_dropout=0.5,
                                                 embedding2nn_type="highway",
                                                 embedding2nn_dim=2048,
                                                 LSTM_output=256,
                                                 LSTM_layers=1,
                                                 max_epoch=100,
                                                 loss_delta_patience=18,
                                                 validation_loss_patience=6,
                                                 metric="macro",
                                                 metric_patience=12,
                                                 verbose=1):
    """
    Trains and evaluates NER models with LOOCV using nested-level mention detection.

    Args:
        files_directory (str): Path to directory containing dataset files.
        model_name (str): Name of the base model for tokenization and embeddings.
        trained_model_directory (str): Directory to save trained models.
        test_files (str/list): Files for LOOCV. Default is "all".
        train_with_validation (float): Fraction of training data used for validation.
        NER_cat_list (list): List of NER categories to train on.
        tagging_scheme (str): Tagging scheme, e.g., "BIO", "BIOES".
        nested_levels (list): List of nested levels for multi-level NER.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        locked_dropout (float): Dropout probability for locked dropout.
        embedding2nn_type (str): Type of embedding-to-NN layer, e.g., "highway".
        embedding2nn_dim (int): Dimension of embedding-to-NN layer.
        LSTM_output (int): LSTM output dimensionality.
        LSTM_layers (int): Number of LSTM layers.
        max_epoch (int): Maximum number of training epochs.
        loss_delta_patience (int): Patience for training loss stagnation.
        validation_loss_patience (int): Patience for validation loss stagnation.
        metric (str): Metric for early stopping, e.g., "macro".
        metric_patience (int): Patience for metric improvement during training.

    Returns:
        None. Saves trained models and metrics as pickle files in `trained_model_directory`.
    """
    # 1. Input Validation
    if NER_cat_list == None:
        raise ValueError(
            f"Error: a NER_cat_list must be provided\nExample: ['PER', 'LOC', 'FAC', 'TIME', 'VEH', 'GPE']")

    # 2. Load Tokenizer and Embedding Model
    # tokenizer, model = load_tokenizer_and_embedding_model(model_name=model_name)
    # embedding_dim = model.config.hidden_size
    config = AutoConfig.from_pretrained(model_name)
    embedding_dim = config.hidden_size

    # 3. Set up model saving directory
    if trained_model_directory == None:
        trained_model_directory = os.path.join(files_directory, f"mentions_detection_model_{model_name.split('/')[-1]}")
    if not os.path.exists(trained_model_directory):
        os.makedirs(trained_model_directory, exist_ok=True)

    # 4. Prepare NER Training Dictionary
    NER_dictionary_name = f"NER_training_dictionary_{model_name.split('/')[-1]}"
    NER_training_dictionary_path = os.path.join(trained_model_directory, f"{NER_dictionary_name}")
    NER_training_dictionary = get_NER_training_dictionary(files_directory,
                                                          NER_training_dictionary_path,
                                                          model_name,
                                                          subword_pooling_strategy=subword_pooling_strategy,
                                                          verbose=verbose)

    # 5. File Splits for Cross-Validation
    if train_files == "all":
        all_files = sorted(list(NER_training_dictionary.keys()))
    if isinstance(train_files, list):
        all_files = [file_name for file_name in train_files if file_name in train_files]

    print(f"Files used in model training: {len(all_files)}")

    if test_files == "all":
        test_files = all_files
    if train_final_model == True:
        test_splits = ["final_model"] + test_files  # the first trained model use all files for training, then cross validation models are trained
    else: test_splits = test_files

    # 6. Identify Unprocessed Files - This allows to resume training
    processed_files = [file for file in os.listdir(trained_model_directory)]

    # 7. Main Training Loop
    for test_split in  tqdm(test_splits, desc="To train models"):
        file_name = str(test_split)[:150]

        if f"{file_name}.pkl" in processed_files:
            print(f"{file_name} already exists, skipping.")
            continue
        trained_model_path = os.path.join(trained_model_directory, f"{file_name}.pkl")
        if test_split in all_files:
            test_files = [test_split]
        elif file_name == 'final_model':
            test_files = []
        else:
            test_files = test_split

        # Train Models for Each Nested Level
        trained_models = []
        for nested_level in nested_levels:
            NER_training_dictionary, id2label, label2id = prepare_NER_dataset(nested_level=nested_level,
                                                                              NER_cat_list=NER_cat_list,
                                                                              tagging_scheme=tagging_scheme,
                                                                              NER_training_dictionary=NER_training_dictionary)

            num_labels = len(label2id)
            print(f"BIOES tags: {num_labels}\nEntities types: {NER_cat_list}\n")
            print(f"Training mention detection model for nested level {nested_level}")

            # Data Preparation
            train_loader, validation_loader = prepare_data_loader(all_files,
                                                                  test_files,
                                                                  NER_training_dictionary=NER_training_dictionary,
                                                                  batch_size=batch_size,
                                                                  train_with_validation=train_with_validation)
            # Model Training
            model, logs = train_NER_model(train_loader=train_loader,
                                          validation_loader=validation_loader,
                                          embedding_dim=embedding_dim,
                                          num_labels=num_labels,  # Number of unique BIOES labels
                                          locked_dropout=locked_dropout,
                                          embedding2nn_type=embedding2nn_type,
                                          embedding2nn_dim=embedding2nn_dim,
                                          LSTM_output=LSTM_output,
                                          LSTM_layers=LSTM_layers,
                                          max_epoch=max_epoch,
                                          learning_rate=learning_rate,
                                          loss_delta_patience=loss_delta_patience,
                                          validation_loss_patience=validation_loss_patience,
                                          metric=metric,
                                          NER_cat_list=NER_cat_list,
                                          id2label=id2label)

            # Store Model Information for current nested level
            trained_models.append({"nested_level": nested_level,
                                   "id2label": id2label,
                                   "label2id": label2id,
                                   "num_labels": num_labels,
                                   "model": model,
                                   "logs": logs})


        models_dict = {}
        for model in trained_models:
            nested_level = model['nested_level']
            trained_model = model['model']
            id2label = model['id2label']
            models_dict[nested_level] = {"model": trained_model,
                                         "id2label": id2label}

        # Models testing loop
        if len(test_files) > 0:
            # List to store inferred entities DataFrames for each test file
            inferred_entities_dfs = []

            for file_name in tqdm(test_files):  # Iterate through all test files
                print(f"Evaluating model on {file_name}")
                # Retrieve token data and gold entity annotations for the current file
                tokens_df = NER_training_dictionary[file_name]["tokens_df"]
                entities_df = NER_training_dictionary[file_name]["entities_df"]

                # Retrieve the test dataset and prepare sentence embeddings for model input
                test_data = NER_training_dictionary[file_name]["dataset"]
                sentences_embeddings = []
                for sentence_data in test_data:  # Collect embeddings for each sentence in the test file
                    sentences_embeddings.append(sentence_data["embeddings"])

                predicted_entities_dfs = []  # List to store predicted entities for all nested levels

                # Iterate through all nested levels of the trained models
                for nested_level in models_dict.keys():
                    # Retrieve the model and label mapping for the current nested level
                    model = models_dict[nested_level]["model"]
                    id2label = models_dict[nested_level]["id2label"]

                    # Perform predictions for the current nested level and get the entities DataFrame
                    predicted_entities_df = get_predicted_entities_df(model, tokens_df, sentences_embeddings, id2label)
                    predicted_entities_dfs.append(predicted_entities_df)  # Append predictions to the list

                # Combine predictions from all nested levels, prioritizing based on confidence scores
                predicted_entities_df = combine_predicted_entities(predicted_entities_dfs, priority="confidence")

                # Combine gold entities and predicted entities for evaluation
                inferred_entities_df = combine_gold_and_predicted_entities_df(entities_df, predicted_entities_df)
                inferred_entities_dfs.append(inferred_entities_df)

            # Concatenate the inferred entities DataFrames from all test files into one
            concatenated_inferred_entities_df = pd.concat(inferred_entities_dfs, axis=0)

            # Calculate evaluation metrics for the combined results
            metrics_df = get_NER_metrics_df(concatenated_inferred_entities_df, NER_cat_list=NER_cat_list)

        else:  # If no test files are provided, set the results to None
            concatenated_inferred_entities_df = None
            metrics_df = None

        # Print the path to the trained model and the evaluation metrics
        print(trained_model_path)
        print(metrics_df)

        # Prepare a dictionary to store all relevant training and evaluation information
        trained_model_infos = {"files_directory": files_directory,
                               "base_model_name": model_name,
                               "all_files": all_files,
                               "test_files": test_files,
                               "tagging_scheme": tagging_scheme,
                               "NER_cat_list": NER_cat_list,
                               "subword_pooling_strategy":subword_pooling_strategy,
                               "train_with_validation": train_with_validation,
                               "batch_size": batch_size,
                               "nested_levels": nested_levels,
                               "embedding_dim": embedding_dim,
                               "locked_dropout": locked_dropout,
                               "embedding2nn_type": embedding2nn_type,
                               "embedding2nn_dim": embedding2nn_dim,
                               "LSTM_output": LSTM_output,
                               "LSTM_layers": LSTM_layers,
                               "max_epoch": max_epoch,
                               "learning_rate": learning_rate,
                               "loss_delta_patience": loss_delta_patience,
                               "validation_loss_patience": validation_loss_patience,
                               "metric": metric,
                               "metric_patience": metric_patience,
                               "models": trained_models,
                               "inferred_entities_df": concatenated_inferred_entities_df,
                               "metrics_df": metrics_df
                               }

        # Save the trained model information and evaluation results to a pickle file
        with open(trained_model_path, "wb") as file:
            pickle.dump(trained_model_infos, file)
            

#%%
def generate_NER_model_card_from_LOOCV_directory(trained_model_directory, model_evaluation_files="all",
                                             read_me_file_name="README"):
    """
    Generates a Hugging Face model card for a trained NER model based on LOOCV (Leave-One-Out Cross-Validation) results.

    Parameters:
        trained_model_directory (str): Directory containing the trained model and evaluation files.
        model_evaluation_files (str or list): Specifies which files to use for evaluation.
            Use "all" to include all files.
        read_me_file_name (str): Name of the README file to save the model card.

    Returns:
        None: Saves the model card as a Markdown file in the specified directory.
    """
    extension = ".pkl"
    trained_model_files = sorted([f.replace(extension, "") for f in os.listdir(trained_model_directory) if f.endswith(extension)])

    if model_evaluation_files != "all":
        trained_model_files = [f for f in trained_model_files if f in model_evaluation_files]


    all_test_files = []
    all_inferred_entities_df = []

    for trained_model_file in tqdm(trained_model_files):
        print(trained_model_file)
        try:
            with open(os.path.join(trained_model_directory, f"{trained_model_file}{extension}"), "rb") as file:
                trained_model_infos = pickle.load(file)

            # Extract key training parameters and architecture details
            foundation_model = trained_model_infos["base_model_name"]
            training_files_directory = trained_model_infos["files_directory"]
            nested_levels = trained_model_infos["nested_levels"]
            embedding_dim = trained_model_infos["embedding_dim"]
            locked_dropout = trained_model_infos["locked_dropout"]
            embedding2nn_type = trained_model_infos["embedding2nn_type"]
            embedding2nn_dim = trained_model_infos["embedding2nn_dim"]
            LSTM_output = trained_model_infos["LSTM_output"]
            labels_number = trained_model_infos['models'][0]['num_labels']
            tagging_scheme = trained_model_infos["tagging_scheme"]
            NER_cat_list = trained_model_infos["NER_cat_list"]
            train_with_validation = trained_model_infos["train_with_validation"]
            batch_size = trained_model_infos["batch_size"]
            learning_rate = trained_model_infos["learning_rate"]
            all_files = trained_model_infos["all_files"]

            test_files = trained_model_infos["test_files"]
            all_test_files.extend(test_files)

            inferred_entities_df = trained_model_infos['inferred_entities_df']

            if len(inferred_entities_df) != 0:
                all_inferred_entities_df.append(inferred_entities_df)
        except:
            pass

    if len(all_inferred_entities_df) == 0:
        model_performances_table = "None"
    else:  # Combine inferred entities across all files
        all_inferred_entities_df = pd.concat(all_inferred_entities_df, axis=0)
        metrics_df = get_NER_metrics_df(all_inferred_entities_df, NER_cat_list=NER_cat_list)

        # Get the macro_avg support value
        macro_avg_support = metrics_df.loc["macro_avg", "support"]
        # Calculate the support ratio for each row
        metrics_df["support %"] = metrics_df["support"] / macro_avg_support

        # Format metrics for readability
        metrics_df["precision"] = metrics_df["precision"].apply(lambda x: f"{x * 100:.2f}%")
        metrics_df["recall"] = metrics_df["recall"].apply(lambda x: f"{x * 100:.2f}%")
        metrics_df["f1_score"] = metrics_df["f1_score"].apply(lambda x: f"{x * 100:.2f}%")
        metrics_df["support"] = metrics_df["support"].apply(lambda x: f"{int(x):,}")
        metrics_df["support %"] = metrics_df["support %"].apply(lambda x: f"{x:.2%}")
        model_performances_table = tabulate(metrics_df, headers="keys", tablefmt="github")

        print(model_performances_table)

    ## Generate a confusion matrix to assess model performance
    def generate_confusion_matrix(inferred_entities_df, NER_cat_list):
        # Extract true and predicted labels
        true_labels = inferred_entities_df['cat_gold']
        predicted_labels = inferred_entities_df['cat_predicted']

        # Generate confusion matrix
        categories = sorted(set(true_labels).union(set(predicted_labels)))
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=categories)

        # Format confusion matrix as a DataFrame
        conf_matrix_df = pd.DataFrame(conf_matrix, index=categories, columns=categories)
        ordered_ner_labels = list(dict(Counter(inferred_entities_df['cat_gold']).most_common()))
        ordered_ner_labels = [label for label in ordered_ner_labels if label in NER_cat_list] + ["O"]
        conf_matrix_df = conf_matrix_df.reindex(ordered_ner_labels)  # Reindex to reorder rows and reset index
        conf_matrix_df = conf_matrix_df[ordered_ner_labels]
        conf_matrix_df["support"] = conf_matrix_df.sum(axis=1)

        # Format numbers for clarity
        for column in ordered_ner_labels + ["support"]:
            conf_matrix_df[column] = conf_matrix_df[column].apply(lambda x: f"{int(x):,}")

        # Add row/column names for clarity
        conf_matrix_df.index.name = "Gold Labels"

        return conf_matrix_df

    # Generate confusion matrix for all predicted mentions
    if len(all_inferred_entities_df) == 0:
        confusion_matrix_table = "None"
    else:
        conf_matrix_df = generate_confusion_matrix(all_inferred_entities_df, NER_cat_list)
        confusion_matrix_table = tabulate(conf_matrix_df, headers="keys", tablefmt="github")


    ## Collect general information about the training corpus
    training_corpus_infos = []
    total_tokens_count = 0
    # Process each training file to count tokens and determine evaluation status
    for file_name in all_files:
        tokens_df = load_tokens_df(file_name, training_files_directory)
        total_tokens_count += len(tokens_df)
        is_in_model_eval = "True" if file_name in all_test_files else "False"
        training_corpus_infos.append({"Document": file_name, "Tokens Count": f"{len(tokens_df):,} tokens",
                                      "Is included in model eval": is_in_model_eval})
    # Summarize training corpus statistics
    training_corpus_infos_df = pd.DataFrame(training_corpus_infos)
    training_corpus_infos_df.loc[len(training_corpus_infos_df), ["Document", "Tokens Count",
                                                                 "Is included in model eval"]] = "TOTAL", f"{total_tokens_count:,} tokens", f"{len(model_evaluation_files)} files used for cross-validation"
    corpus_infos_table = tabulate(training_corpus_infos_df, headers="keys", tablefmt="github")

    ## GENERATE HF MODEL CARD READ.ME FILE
    read_me = f"""
---
language: fr
tags:
- NER
- camembert
- literary-texts
- nested-entities
- propp-fr
license: apache-2.0
metrics:
- f1
- precision
- recall
base_model:
- {foundation_model}
pipeline_tag: token-classification
---

## INTRODUCTION:
This model, developed as part of the [propp-fr project](https://lattice-8094.github.io/propp/), is a **NER model** built on top of [{foundation_model.split("/")[-1]}](https://huggingface.co/{foundation_model}) embeddings, trained to predict nested entities in french, specifically for literary texts.

The predicted entities are:
- mentions of characters (PER): pronouns (je, tu, il, ...), possessive pronouns (mon, ton, son, ...), common nouns (le capitaine, la princesse, ...) and proper nouns (Indiana Delmare, Honoré de Pardaillan, ...)
- facilities (FAC): chatêau, sentier, chambre, couloir, ...
- time (TIME): le règne de Louis XIV, ce matin, en juillet, ...
- geo-political entities (GPE): Montrouge, France, le petit hameau, ...
- locations (LOC): le sud, Mars, l'océan, le bois, ...
- vehicles (VEH): avion, voitures, calèche, vélos, ...

## MODEL PERFORMANCES (LOOCV):
{model_performances_table}

## TRAINING PARAMETERS:
- Entities types: {NER_cat_list}
- Tagging scheme: {tagging_scheme}
- Nested entities levels: {nested_levels}
- Split strategy: Leave-one-out cross-validation ({len(training_corpus_infos)} files)
- Train/Validation split: {train_with_validation} / {1 - train_with_validation:.2f}
- Batch size: {batch_size}
- Initial learning rate: {learning_rate}

## MODEL ARCHITECTURE:
Model Input: Maximum context {foundation_model.split("/")[-1]} embeddings ({embedding_dim} dimensions)

- Locked Dropout: {locked_dropout}

- Projection layer:
  - layer type: {embedding2nn_type} layer
  - input: {embedding_dim} dimensions
  - output: {embedding2nn_dim} dimensions

- BiLSTM layer:
  - input: {embedding2nn_dim} dimensions
  - output: {LSTM_output} dimensions (hidden state)

- Linear layer:
  - input: {LSTM_output} dimensions
  - output: {labels_number} dimensions (predicted labels with {tagging_scheme} tagging scheme)

- CRF layer

Model Output: {tagging_scheme} labels sequence

## HOW TO USE:
[Propp Documentation](https://lattice-8094.github.io/propp/quick_start/)

## TRAINING CORPUS:
{corpus_infos_table}

## PREDICTIONS CONFUSION MATRIX:
{confusion_matrix_table}

## CONTACT:
mail: antoine [dot] bourgois [at] protonmail [dot] com
"""

    ## Save the model card as a Markdown file
    output_path = os.path.join(trained_model_directory, f"{read_me_file_name}.md")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(read_me)

    print(f"model card {read_me_file_name}.md file has been saved at {output_path}")
    
    

## Mentions Detection Model Inference
import requests
def load_mentions_detection_model(model_path="AntoineBourgois/propp-fr_NER_camembert-large_FAC_GPE_LOC_PER_TIME_VEH", force_download=False):
    """
    Loads a mentions detection model from a specified path. It first checks for the model locally,
    and if not found (or if force_download is True), downloads it from HuggingFace.

    Args:
        model_path (str): Path to the model. This can be a local file, directory, or a HuggingFace model name.
        force_download (bool): If True, the mentions detection model is downloaded from HuggingFace
                               and the local model is overwritten.

    Returns:
        object: The loaded mentions detection model.

    Raises:
        requests.exceptions.RequestException: For HTTP errors while downloading the model.
        pickle.UnpicklingError: For errors during model deserialization.
        FileNotFoundError: If the specified local file or directory doesn't exist.
        Exception: For other unexpected errors.
    """

    def download_model_from_huggingface():
        """Downloads the model from HuggingFace and saves it locally."""
        print(f"Downloading model from HuggingFace: https://huggingface.co/{model_path}")
        url_model_path = f"https://huggingface.co/{model_path}/resolve/main/final_model.pkl"

        try:
            response = requests.get(url_model_path)
            response.raise_for_status()  # Raise an exception for HTTP errors
            mentions_detection_model = pickle.loads(response.content)  # Deserialize the downloaded model
            print("Model Downloaded Successfully")

            # Save the model locally for future use
            mentions_models_directory = os.path.dirname(local_mentions_detection_model_path)
            absolute_directory = os.path.abspath(mentions_models_directory)
            if not os.path.exists(absolute_directory):
                os.makedirs(absolute_directory)

            print(f"Saving model locally to: {absolute_directory}")
            with open(local_mentions_detection_model_path, "wb") as file:
                pickle.dump(mentions_detection_model, file)

            return mentions_detection_model

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
    local_mentions_detection_model_path = os.path.join(current_directory, f"{model_path}/final_model.pkl")

    try:
        if not force_download:
            # 1. Attempt to load model from the given file path
            if os.path.isfile(model_path):  # Check if the provided path is a file
                with open(model_path, "rb") as file:
                    mentions_detection_model = pickle.load(file)
                print(f"Model Loaded Successfully from local path: {model_path}")
                return mentions_detection_model

            # 2. Attempt to load model from the "final_model.pkl" in the current working directory
            if os.path.exists(local_mentions_detection_model_path):
                with open(local_mentions_detection_model_path, "rb") as file:
                    mentions_detection_model = pickle.load(file)
                print(f"Model Loaded Successfully from local path: {local_mentions_detection_model_path}")
                return mentions_detection_model

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

def predict_BIOES_tags(model, sentences_embeddings, batch_size=32, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the model to evaluation mode
    model.eval()

    # Prepare batched inference
    num_sentences = len(sentences_embeddings)
    original_lengths = [sentence.size(0) for sentence in sentences_embeddings]  # Track original lengths
    all_predictions = []
    all_confidences = []

    if verbose:
        for i in tqdm(range(0, num_sentences, batch_size), desc="Predicting BIOES tags", leave=False):
            # Get the current batch of sentence embeddings
            batch_embeddings = sentences_embeddings[i:i + batch_size]

            # Pad sentences to the same length
            batch_embeddings_padded = pad_sequence(batch_embeddings,
                                                   batch_first=True)  # Shape: (batch_size, max_seq_len, embedding_dim)

            # Create attention masks (1 for real tokens, 0 for padding)
            attention_masks = torch.zeros(batch_embeddings_padded.size()[:2],
                                          dtype=torch.long)  # Shape: (batch_size, max_seq_len)
            for j, sentence in enumerate(batch_embeddings):
                attention_masks[j, :sentence.size(0)] = 1  # Set real tokens to 1

            # Move tensors to the correct device
            batch_embeddings_padded = batch_embeddings_padded.to(device)
            attention_masks = attention_masks.to(device)

            # Perform inference
            with torch.no_grad():
                predictions, confidences = model(batch_embeddings_padded, attention_mask=attention_masks)

            for j, length in enumerate(original_lengths[i:i + batch_size]):
                # Extract only the unpadded predictions and confidences
                all_predictions.append(predictions[j][:length])
                all_confidences.append(confidences[j][:length])
    else:
        for i in range(0, num_sentences, batch_size):
            # Get the current batch of sentence embeddings
            batch_embeddings = sentences_embeddings[i:i + batch_size]

            # Pad sentences to the same length
            batch_embeddings_padded = pad_sequence(batch_embeddings,
                                                   batch_first=True)  # Shape: (batch_size, max_seq_len, embedding_dim)

            # Create attention masks (1 for real tokens, 0 for padding)
            attention_masks = torch.zeros(batch_embeddings_padded.size()[:2],
                                          dtype=torch.long)  # Shape: (batch_size, max_seq_len)
            for j, sentence in enumerate(batch_embeddings):
                attention_masks[j, :sentence.size(0)] = 1  # Set real tokens to 1

            # Move tensors to the correct device
            batch_embeddings_padded = batch_embeddings_padded.to(device)
            attention_masks = attention_masks.to(device)

            # Perform inference
            with torch.no_grad():
                predictions, confidences = model(batch_embeddings_padded, attention_mask=attention_masks)

            for j, length in enumerate(original_lengths[i:i + batch_size]):
                # Extract only the unpadded predictions and confidences
                all_predictions.append(predictions[j][:length])
                all_confidences.append(confidences[j][:length])

    return all_predictions, all_confidences

def generate_entities_df(tokens_df, tokens_embedding_tensor, mentions_detection_model, batch_size=32, verbose=False):
    """
    Generates a DataFrame containing predicted entities from tokens and their embeddings.

    Args:
        tokens_df (pd.DataFrame): DataFrame with tokens and their metadata, including sentence IDs.
        tokens_embedding_tensor (torch.Tensor): Tensor containing token embeddings.
        mentions_detection_model (dict): Model and metadata for entity detection, including multiple NER models.
        batch_size (int, optional): Batch size for model inference. Defaults to 32.

    Returns:
        pd.DataFrame: DataFrame containing predicted entities, their token indices, confidence scores, and text.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Step 1: Prepare embeddings for each sentence
    sentences_embeddings = []
    for sentence_ID, sentence_tokens_df in tokens_df.groupby("sentence_ID"):
        tokens_ids = list(sentence_tokens_df.index)
        sentence_embeddings = tokens_embedding_tensor[tokens_ids]
        sentence_embeddings = sentence_embeddings.to(dtype=torch.float32)
        # Ensure each sentence's embeddings have two dimensions, even if it contains only one token
        if sentence_embeddings.dim() == 1:
            sentence_embeddings = sentence_embeddings.unsqueeze(0)
        sentences_embeddings.append(sentence_embeddings)

    # Step 2: Predict entities using each nested level NER model in the mentions_detection_model
    predicted_entities_dfs = []
    if verbose:
        print("Predicting BIOES tags for nested levels...")
    for NER_model in mentions_detection_model["models"]:
        model = NER_model["model"].to(device) # Extract the NER model
        id2label = NER_model["id2label"] # Mapping from IDs to BIOES labels

        # Predict BIOES tags and confidences for all sentences in batches
        all_predictions, all_confidences = predict_BIOES_tags(model, sentences_embeddings, batch_size=batch_size, verbose=verbose)

        # Convert prediction IDs to BIOES labels
        predicted_BIO_tags = [
            [id2label[id] for id in sequence]
            for sequence in all_predictions
        ]

        # Remove illegal transitions in BIOES tags
        legal_predicted_BIO_tags = [remove_BIO_illegal_transitions(sentence) for sentence in predicted_BIO_tags]

        # Flatten the BIOES tags to prepare for entity extraction
        flatten_legal_predicted_BIO_tags_predictions = [label for sentence in legal_predicted_BIO_tags for label in
                                                        sentence]

        # Extract entities based on BIOES tag sequences
        predicted_entities_df = extract_entities_from_BIO_tag_list(flatten_legal_predicted_BIO_tags_predictions)

        # Flatten the confidence scores for consistency with the flattened tag list
        confidences = [confidence for sequence in all_confidences for confidence in sequence]

        # Calculate mean confidence for each predicted entity
        mentions_confidence = []
        for start_token, end_token in predicted_entities_df[['start_token', 'end_token']].values:
            confidence_values = confidences[start_token:end_token + 1]
            mean_confidence = sum(confidence_values) / len(confidence_values)
            mentions_confidence.append(mean_confidence)

        # Add confidence scores to the predicted entities DataFrame
        predicted_entities_df['confidence'] = mentions_confidence
        # Append the current model's results to the list of predicted entities DataFrames
        predicted_entities_dfs.append(predicted_entities_df)

    # Step 3: Combine predictions from all models, prioritizing those with higher confidence
    predicted_entities_df = combine_predicted_entities(predicted_entities_dfs, priority="confidence")

    # Step 4: Add the corresponding text for each predicted entity
    mentions_text = []
    for start_token, end_token in predicted_entities_df[["start_token", "end_token"]].values:
        text = " ".join(list(tokens_df.loc[start_token:end_token, "word"]))
        mentions_text.append(text)
    predicted_entities_df["text"] = mentions_text

    return predicted_entities_df
