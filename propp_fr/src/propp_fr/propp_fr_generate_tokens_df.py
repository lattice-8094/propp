import subprocess
import sys
from tqdm.auto import tqdm
import gc
import pandas as pd
import torch
import spacy


## Get tokens_df from text_file
def load_spacy_model(model_name='fr_dep_news_trf', model_max_length=500000):

    # Loading Spacy French Transformer Model and Enabling GPU
    spacy.prefer_gpu()

    try:
        model = spacy.load(model_name)
    except OSError:
        subprocess.run([sys.executable, '-m', 'spacy', 'download', model_name], check=True)
        model = spacy.load(model_name)

    model.max_length = model_max_length
    print(f'Loaded Spacy Model: {model_name}')

    # Ensure that the model's transformer is using the GPU if available
    if torch.cuda.is_available():
        print("CUDA is available, model should run on GPU.")
    else:
        print("CUDA is not available, model will run on CPU.")

    return model

def generate_tokens_df_from_spacy_doc(doc):
    token_dict = []
    paragraph_ID = 0
    sentence_ID = -1
    previous_is_newline_char = True
    previous_is_sent_start = False
    previous_is_punct = False
    token_ID_within_sentence = 0

    for token in doc:
        word = token.text
        lemma = token.lemma_
        token_ID_within_document = token.i
        byte_onset = token.idx
        byte_offset = token.idx + len(word)
        is_newline_character = '\n' in token.text_with_ws
        POS_tag = token.pos_
        is_sent_start = token.is_sent_start
        if previous_is_newline_char or ((
                                                token.is_title or token.is_punct) and is_sent_start and previous_is_punct and not previous_is_sent_start):
            is_sent_start = True
        else:
            is_sent_start = False
        dependency_relation = token.dep_
        syntactic_head_ID = token.head.i

        if word in ['.', '!', '?']:
            previous_is_punct = True
        else:
            previous_is_punct = False

        if is_sent_start:
            previous_is_sent_start = True
            sentence_ID += 1
            token_ID_within_sentence = 0
        else:
            previous_is_sent_start = False
            token_ID_within_sentence += 1

        if is_newline_character == True:
            paragraph_ID += 1
            previous_is_newline_char = True
        else:
            previous_is_newline_char = False
            token_dict.append({'paragraph_ID': paragraph_ID,
                               'sentence_ID': sentence_ID,
                               'token_ID_within_sentence': token_ID_within_sentence,
                               'token_ID_within_document': token_ID_within_document,
                               'word': word,
                               'lemma': lemma,
                               'byte_onset': byte_onset,
                               'byte_offset': byte_offset,
                               'POS_tag': POS_tag,
                               'dependency_relation': dependency_relation,
                               'syntactic_head_ID': syntactic_head_ID,
                               })

    tokens_df = pd.DataFrame(token_dict)
    del doc
    gc.collect()
    torch.cuda.empty_cache()

    token_id_mapping = {}
    for i, previous_token_id in enumerate(tokens_df['token_ID_within_document'].tolist()):
        token_id_mapping[previous_token_id] = i
    # Replace IDs in 'token_ID_within_document' column
    tokens_df['token_ID_within_document'] = tokens_df['token_ID_within_document'].map(token_id_mapping).fillna(
        tokens_df['token_ID_within_document']).astype(int)
    # Replace IDs in 'syntactic_head_ID' column
    tokens_df['syntactic_head_ID'] = tokens_df['syntactic_head_ID'].map(token_id_mapping).fillna(
        tokens_df['syntactic_head_ID']).astype(int)

    return tokens_df

def generate_tokens_df(text_content, spacy_model, max_char_sentence_length=100000, verbose=1):
    text_len = len(text_content)
    sample_count = (text_len // max_char_sentence_length) + 1
    sample_boundaries = [i for i in range(0, text_len, text_len // sample_count)] + [text_len]

    tokens_df = pd.DataFrame()
    start_boundary = 0
    for end_boundary in tqdm(sample_boundaries[1:], desc='Batch Spacy Tokenization', leave=False, disable=(verbose == 0)):
        sample_text = text_content[start_boundary:end_boundary]
        sample_doc = spacy_model(sample_text)
        sample_tokens_df = generate_tokens_df_from_spacy_doc(sample_doc)

        if len(tokens_df) == 0:
            max_sentence_id = sample_tokens_df['sentence_ID'].max()
            last_sentence_start_row = sample_tokens_df[sample_tokens_df['sentence_ID'] == max_sentence_id].iloc[0]
            if end_boundary != sample_boundaries[-1]:
                sample_tokens_df = sample_tokens_df.iloc[:last_sentence_start_row.name]
            tokens_df = sample_tokens_df

        else:
            sample_tokens_df['paragraph_ID'] = sample_tokens_df['paragraph_ID'] + previous_last_sentence_start_row[
                'paragraph_ID']
            sample_tokens_df['sentence_ID'] = sample_tokens_df['sentence_ID'] + previous_last_sentence_start_row[
                'sentence_ID']
            sample_tokens_df['token_ID_within_document'] = sample_tokens_df['token_ID_within_document'] + \
                                                           previous_last_sentence_start_row['token_ID_within_document']
            sample_tokens_df['byte_onset'] = sample_tokens_df['byte_onset'] + previous_last_sentence_start_row[
                'byte_onset']
            sample_tokens_df['byte_offset'] = sample_tokens_df['byte_offset'] + previous_last_sentence_start_row[
                'byte_onset']
            sample_tokens_df['syntactic_head_ID'] = sample_tokens_df['syntactic_head_ID'] + \
                                                    previous_last_sentence_start_row['token_ID_within_document']

            max_sentence_id = sample_tokens_df['sentence_ID'].max()
            last_sentence_start_row = sample_tokens_df[sample_tokens_df['sentence_ID'] == max_sentence_id].iloc[0]
            if end_boundary != sample_boundaries[-1]:
                sample_tokens_df = sample_tokens_df.iloc[:last_sentence_start_row.name]

            tokens_df = pd.concat([tokens_df, sample_tokens_df], ignore_index=True)

        previous_last_sentence_start_row = last_sentence_start_row
        start_boundary = previous_last_sentence_start_row['byte_onset']

        del sample_doc
        gc.collect()
        torch.cuda.empty_cache()

    return tokens_df
