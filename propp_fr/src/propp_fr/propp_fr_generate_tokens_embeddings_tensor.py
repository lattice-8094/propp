from transformers import AutoTokenizer, AutoModel, MT5EncoderModel, AutoTokenizer, T5EncoderModel
import torch
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import gc

#def load_tokenizer_and_embedding_model(model_name="almanach/camembert-large"):
#
#    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
#    print(f"Tokenizer and Embedding Model Initialized: {model_name}")
#
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    model.to(device)
#
#    return tokenizer, model
    
def load_tokenizer_and_embedding_model(model_name):
    """
    Loads tokenizer and embedding model with hidden states enabled.
    Supports encoder-only transformers and T5 (encoder).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "mt5" in model_name.lower():
        # T5 is encoder-decoder → we ONLY want the encoder
        model = MT5EncoderModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        print(f"Loaded MT5 encoder: {model_name}")
    elif "t5" in model_name.lower():
        # T5 is encoder-decoder → we ONLY want the encoder
        model = T5EncoderModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        print(f"Loaded T5 encoder: {model_name}")
    elif "ul2" in model_name.lower():
        model = T5EncoderModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=torch.float16,
        )
        print(f"Loaded ul2 encoder: {model_name}")
    else:
        model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        print(f"Loaded encoder model: {model_name}")

    model.to(device)
    model.eval()  # IMPORTANT for embedding extraction

    return tokenizer, model
    

def tokenize_text(text, tokenizer):
    """Tokenizes text and retrieves token IDs with byte offsets"""
    encoding = tokenizer(text,
                         return_offsets_mapping=True,
                         return_tensors="pt",
                         return_attention_mask=False,
                         add_special_tokens=False)

    subword_ids = encoding["input_ids"].squeeze().tolist()  # Token IDs
    subword_offsets = encoding["offset_mapping"].squeeze().tolist()  # (Start, End) byte positions
    subwords = tokenizer.convert_ids_to_tokens(subword_ids)  # Convert IDs to readable tokens

    return subwords, subword_ids, subword_offsets

def find_subwords_for_tokens(tokens_df, subword_offsets):
    subword_onsets = np.array([offset[0] for offset in subword_offsets])
    subword_offsets = np.array([offset[1] for offset in subword_offsets])

    tokens_subword_ids = []

    for token_byte_onset, token_byte_offset in tokens_df[["byte_onset", "byte_offset"]].values:
        # Find the first subword that starts at or before the token onset
        start_idx = np.searchsorted(subword_onsets, token_byte_onset, side="right") - 1
        start_idx = max(start_idx, 0)  # Ensure we don't go below index 0

        # Expand right while ensuring subwords still overlap with the token
        end_idx = start_idx
        while end_idx < len(subword_offsets) and subword_onsets[end_idx] < token_byte_offset:
            end_idx += 1

        # Allow subwords that start before the token but overlap significantly
        contained_subwords_ids = [
            i for i in range(start_idx, end_idx)
            if not (subword_offsets[i] <= token_byte_onset or subword_onsets[i] >= token_byte_offset)
        ]

        tokens_subword_ids.append(contained_subwords_ids)

    return tokens_subword_ids

def get_boudaries_list(tokens_df, sub_word_offsets, sliding_window_size=0, sliding_window_overlap=0.5):
    import numpy as np

    token_offsets = np.array(tokens_df["byte_offset"].tolist())
    token_offset_set = set(token_offsets)

    possible_end_tokens = np.array([i for i, (_, end) in enumerate(sub_word_offsets) if end in token_offset_set])
    possible_start_tokens = np.array([0] + (possible_end_tokens + 1).tolist())

    overlapping_window_boundaries = []
    window_start = 0
    prev_window_start = -1

    while window_start < len(sub_word_offsets):
        if window_start == prev_window_start:
            print(f"No progress detected at window_start={window_start} — breaking to avoid infinite loop.")
            break
        prev_window_start = window_start

        if window_start + sliding_window_size >= len(sub_word_offsets):
            window_end = len(sub_word_offsets)
            overlapping_window_boundaries.append((window_start, window_end))
            break

        max_idx = np.searchsorted(possible_end_tokens, window_start + sliding_window_size, side='right') - 1
        window_end = possible_end_tokens[max_idx] if max_idx >= 0 else window_start

        overlapping_window_boundaries.append((window_start, window_end))

        # Calculate next start with overlap
        new_start_pos = window_end - int(sliding_window_size * sliding_window_overlap)
        start_idx = np.searchsorted(possible_start_tokens, new_start_pos, side='right') - 1

        next_window_start = possible_start_tokens[start_idx] if start_idx >= 0 else window_start + 1

        # Force forward progress
        if next_window_start <= window_start:
            next_window_start = window_start + 1

        window_start = next_window_start

    return overlapping_window_boundaries

def compute_sub_word_embeddings(boundaries_list, token_ids, model, mini_batch_size=10, padding_token_id=0, sliding_window_size=0, device='cpu', verbose=1):

    # Convert token_ids to a single tensor before the loop
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    max_token_id = token_ids.size(0)

    # Precompute the padding tensor once
    padding_tensor = torch.tensor([padding_token_id] * sliding_window_size, dtype=torch.long).unsqueeze(0).to(device)

    batch_input_ids = []
    all_embeddings_batches = []

    with torch.no_grad():
        for start_boundary, end_boundary in tqdm(boundaries_list, desc='Embedding Tokens', leave=False, disable=(verbose == 0)):
            input_ids = token_ids[start_boundary:end_boundary]
            real_tokens_length = input_ids.size(0)

            # Pad only if needed
            if real_tokens_length < sliding_window_size:
                padded_input = torch.cat([input_ids.unsqueeze(0), padding_tensor[:, real_tokens_length:]], dim=1)
            else:
                padded_input = input_ids.unsqueeze(0)


            batch_input_ids.append(padded_input)

            # Process the batch based on the specified conditions
            if (len(batch_input_ids) == mini_batch_size) or (
                    end_boundary >= max_token_id):  # Ensure end_boundary is a single value
                # Process the batch
                batch_input_ids_tensor = torch.cat(batch_input_ids,
                                                   dim=0)  # Concatenate all input tensors along the batch dimension
                attention_mask = (batch_input_ids_tensor != padding_token_id).long()  # Create attention mask

                # Move to device if not already
                batch_input_ids_tensor = batch_input_ids_tensor.to(device)
                attention_mask = attention_mask.to(device)

                # Get model outputs
                outputs = model(batch_input_ids_tensor, attention_mask=attention_mask)
                last_hidden_states = outputs.last_hidden_state  # Get last hidden states

                # Reshape last hidden states
                last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[2])

                # Flatten hidden states and filter out padding
                token_embeddings = last_hidden_states[attention_mask.view(-1) == 1]

                all_embeddings_batches.append(token_embeddings.cpu())

                # Reset the batch
                batch_input_ids = []

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings_batches)

    subword_indices = []
    for start_boundary, end_boundary in boundaries_list:
        subword_indices += list(range(start_boundary, end_boundary))

    return subword_indices, all_embeddings

def average_embeddings_from_overlapping_sliding_windows(subword_indices, all_embeddings):
    subword_indices = torch.tensor(subword_indices, device=all_embeddings.device)  # Convert to tensor
    unique_indices, inverse_indices = torch.unique(subword_indices, return_inverse=True)  # Get unique indices

    # Get counts for each unique subword index
    counts = torch.bincount(inverse_indices, minlength=len(unique_indices)).float().unsqueeze(1)

    # Use bincount to sum embeddings efficiently
    summed_embeddings = torch.zeros((len(unique_indices), all_embeddings.shape[1]), device=all_embeddings.device)
    summed_embeddings.index_add_(0, inverse_indices, all_embeddings)

    # Compute mean embeddings
    mean_embeddings = summed_embeddings / counts

    return mean_embeddings

def get_token_embeddings(mean_subword_embeddings,
                         tokens_subword_ids,
                         subword_pooling_strategy="average" # ["average", "first", "last", "first_last"]
                         ):

    """
    Pool subword embeddings to get token-level embeddings, with handling for empty token-subword lists.

    Args:
        mean_subword_embeddings (torch.Tensor): Tensor containing subword embeddings (num_subwords, embedding_dim).
        tokens_subword_ids (list of lists): List of lists where each list contains the indices of the subwords
                                            corresponding to a token.
        subword_pooling_strategy (str): The pooling strategy to use for merging subword embeddings. Options are:
                                       "average", "first", "last", "first_last".

    Returns:
        torch.Tensor: Token-level embeddings with shape (len(tokens_subword_ids), embedding_dim)
    """

    # Pre-allocate zero tensor using the shape of the first subword embedding
    zero_tensor = torch.zeros(mean_subword_embeddings.shape[1])

    # Initialize an empty tensor for token embeddings (same length as number of tokens)
    token_embeddings = torch.zeros(len(tokens_subword_ids), mean_subword_embeddings.shape[1])

    for idx, token_subword_ids in enumerate(tokens_subword_ids):
        if len(token_subword_ids) == 0:
            token_embeddings[idx] = zero_tensor
        else:
            # Extract the subword embeddings corresponding to the current token
            token_subword_embeddings = mean_subword_embeddings[token_subword_ids]

            # Apply pooling strategy
            if subword_pooling_strategy == "average":
                # Average the embeddings of all subwords
                token_embeddings[idx] = token_subword_embeddings.mean(dim=0)
            elif subword_pooling_strategy == "first":
                # Take the first subword embedding
                token_embeddings[idx] = token_subword_embeddings[0]
            elif subword_pooling_strategy == "last":
                # Take the last subword embedding
                token_embeddings[idx] = token_subword_embeddings[-1]
            elif subword_pooling_strategy == "first_last":
                # Average the first and last subword embeddings
                token_embeddings[idx] = (token_subword_embeddings[0] + token_subword_embeddings[-1]) / 2
            elif subword_pooling_strategy == "max":
            	# Take the maximum value across subwords
            	token_embeddings[idx] = token_subword_embeddings.max(dim=0)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {subword_pooling_strategy}")

    return token_embeddings

def get_embedding_tensor_from_tokens_df(text,
                                        tokens_df,
                                        tokenizer,
                                        model,
                                        sliding_window_size='max',
                                        mini_batch_size=12,
                                        sliding_window_overlap=0.5,
                                        subword_pooling_strategy="average", # ["average", "first", "last", "first_last", "max"]
                                        device=None,
                                        verbose=1):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sliding_window_size=='max':
        if tokenizer.model_max_length > 100000:
            sliding_window_size = 512
        else:
            sliding_window_size = tokenizer.model_max_length


    padding_token_id = int(tokenizer.pad_token_id)
    subwords, subword_ids, subword_offsets = tokenize_text(text, tokenizer)
    tokens_subword_ids = find_subwords_for_tokens(tokens_df, subword_offsets)
    overlapping_window_boundaries = get_boudaries_list(tokens_df, sub_word_offsets=subword_offsets, sliding_window_size=sliding_window_size, sliding_window_overlap=sliding_window_overlap)
    subword_indices, all_subword_embeddings = compute_sub_word_embeddings(overlapping_window_boundaries, subword_ids, model, mini_batch_size=mini_batch_size, padding_token_id=padding_token_id, sliding_window_size=sliding_window_size, device=device, verbose=verbose)
    mean_subword_embeddings = average_embeddings_from_overlapping_sliding_windows(subword_indices, all_subword_embeddings)
    del all_subword_embeddings
    gc.collect()
    tokens_embeddings_tensor = get_token_embeddings(mean_subword_embeddings,
                                                    tokens_subword_ids,
                                                    subword_pooling_strategy=subword_pooling_strategy # ["average", "first", "last", "first_last"]
                             )

    # tokens_df["subword_ids"] = tokens_subword_ids
    # tokens_df["subword_offsets"] = tokens_df["subword_ids"].apply(lambda subword_ids: [subword_offsets[i] for i in subword_ids])
    # tokens_df["subwords"] = tokens_df["subword_ids"].apply(lambda subword_ids: [subwords[i] for i in subword_ids])


    return tokens_embeddings_tensor

