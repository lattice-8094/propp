from tqdm.auto import tqdm
import os

def reconstruct_text_from_tokens_df(tokens_df):
    """
    Reconstructs the original text from a DataFrame of tokens, preserving paragraph breaks
    and ensuring proper alignment with byte offsets.

    Args:
        tokens_df (DataFrame): DataFrame containing the tokenized text with columns:
                               'paragraph_ID', 'word', 'byte_onset', 'byte_offset'.

    Returns:
        str: Reconstructed text as a single string with paragraph breaks.
    """
    current_char_id = 0
    current_paragraph_ID = 0
    generated_text = ""

    for paragraph_ID, word, byte_onset, byte_offset in tokens_df[
        ["paragraph_ID", "word", "byte_onset", "byte_offset"]].values:
        word = str(word)

        # Add a newline for each new paragraph
        if paragraph_ID > current_paragraph_ID:
            generated_text += "\n"
            current_char_id += 1
            current_paragraph_ID = paragraph_ID

        # Add spaces to align with the byte offset
        while current_char_id < byte_onset:
            generated_text += " "
            current_char_id += 1

        # Append the word and update the current character ID
        generated_text += word
        current_char_id = byte_offset

    return generated_text

def generating_sacr_annotations(raw_text,
                                tokens_df,
                                entities_df,
                                entity_type_column="cat",
                                coref_name_column="COREF"):
    """
    Generates SACR-compatible annotations by embedding inline entity mentions and coreference
    chains within the raw text.

    Args:
        raw_text (str): The original reconstructed text.
        tokens_df (DataFrame): DataFrame with tokenized text, including byte offsets.
        entities_df (DataFrame): DataFrame with entity mentions and their metadata, including:
                                 'start_token', 'end_token', 'mention_len', and the columns
                                 specified by `entity_type_column` and `coref_name_column`.
        entity_type_column (str): Column name for entity type (default: "cat").
        coref_name_column (str): Column name for coreference ID (default: "COREF").

    Returns:
        str: Annotated text with inline SACR-compatible annotations.
    """
    annotated_text = raw_text

    # Sort entities by length of mention and start position for proper insertion order
    entities_df = entities_df.sort_values(by=['mention_len', 'start_token'], ascending=[True, True]).reset_index(
        drop=True)

    for COREF_name, start_token, end_token, cat, text in tqdm(
            entities_df[[coref_name_column, 'start_token', 'end_token', entity_type_column, 'text']].values,
            desc="Generating Sacr Annotations"):
        # Retrieve byte offsets for the mention
        origin_mention_byte_onset = tokens_df.loc[start_token, 'byte_onset']
        origin_mention_byte_offset = tokens_df.loc[end_token, 'byte_offset']

        # Construct the inline annotation
        mention_prefix = f'{{{COREF_name}:EN="{cat}" '
        mention_suffix = '}'
        annotated_entity = mention_prefix + annotated_text[
                                            origin_mention_byte_onset:origin_mention_byte_offset] + mention_suffix

        # Adjust token offsets to account for annotation length changes
        tokens_df["temp_byte_onset"], tokens_df["temp_byte_offset"] = tokens_df["byte_onset"], tokens_df["byte_offset"]

        # Update byte_onset and byte_offset for tokens after the annotation insertion point
        byte_onset_to_change = tokens_df[tokens_df["byte_onset"] > origin_mention_byte_onset]
        tokens_df.loc[byte_onset_to_change.index, 'temp_byte_onset'] = byte_onset_to_change['temp_byte_onset'] + len(
            mention_prefix)

        byte_onset_to_change = tokens_df[tokens_df["byte_onset"] >= origin_mention_byte_onset]
        tokens_df.loc[byte_onset_to_change.index, 'temp_byte_offset'] = byte_onset_to_change['temp_byte_offset'] + len(
            mention_prefix)

        byte_onset_to_change = tokens_df[tokens_df["byte_offset"] >= origin_mention_byte_offset]
        tokens_df.loc[byte_onset_to_change.index, 'temp_byte_offset'] = byte_onset_to_change['temp_byte_offset'] + len(
            mention_suffix)

        byte_onset_to_change = tokens_df[tokens_df["byte_offset"] > origin_mention_byte_offset]
        tokens_df.loc[byte_onset_to_change.index, 'temp_byte_onset'] = byte_onset_to_change['temp_byte_onset'] + len(
            mention_suffix)

        # Apply the updated byte offsets
        tokens_df["byte_onset"], tokens_df["byte_offset"] = tokens_df["temp_byte_onset"], tokens_df["temp_byte_offset"]

        # Insert the annotated entity into the text
        annotated_text = annotated_text[:origin_mention_byte_onset] + annotated_entity + annotated_text[
                                                                                         origin_mention_byte_offset:]

    return annotated_text

def generate_sacr_file(file_name,
                       tokens_df,
                       entities_df,
                       end_directory,
                       entity_type_column="cat",
                       coref_name_column="COREF",
                       sacr_extension=".generated_sacr"):
    """
    Generates a SACR file with inline annotations for mention detection and coreference chains.

    Args:
        file_name (str): Base name for the output file (without extension).
        tokens_df (DataFrame): DataFrame with tokenized text, including byte offsets.
        entities_df (DataFrame): DataFrame with entity metadata.
        end_directory (str): Directory to save the generated SACR file.
        entity_type_column (str): Column name for entity type (default: "cat").
        coref_name_column (str): Column name for coreference ID (default: "COREF").
        sacr_extension (str): File extension for the SACR file (default: ".generated_sacr").

    Returns:
        None: Saves the annotated SACR file to the specified directory.
    """
    # Reconstruct the original raw text from tokens
    raw_text = reconstruct_text_from_tokens_df(tokens_df)

    # Generate the SACR-compatible annotated text
    sacr_annotated_text = generating_sacr_annotations(raw_text,
                                                      tokens_df,
                                                      entities_df,
                                                      entity_type_column=entity_type_column,
                                                      coref_name_column=coref_name_column)

    import re
    # Add additional spacing for paragraph formatting
    sacr_annotated_text = sacr_annotated_text.replace("\n", "\n\n")
    # Replace multiple spaces (but not newlines) with a single space
    sacr_annotated_text = re.sub(r'(?<=\S) {2,}(?=\S)', ' ', sacr_annotated_text)
    sacr_annotated_text = sacr_annotated_text.replace("\n\n ", "\n\n")
    sacr_annotated_text = sacr_annotated_text.replace("\n ", "\n")
    sacr_annotated_text = sacr_annotated_text.replace("\n ", "\n")

    # Save the generated SACR file
    file_path = os.path.join(end_directory, f"{file_name}{sacr_extension}")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(sacr_annotated_text)

    print(f"File saved at:\n{file_path}")

    return sacr_annotated_text