---
title: "Quick Start"
---

**Getting Started with French Text Processing**

This guide will help you load and analyze French text files using the `propp_fr` library.

## Installation

The French variant of the Propp python library can be installed via [pypi](https://pypi.org/project/propp-fr/):

```bash
pip install propp_fr
```

## Oneliner Processing

You can process a text file in one line with the default models:

```python
from propp_fr import process_text_file

process_text_file("root_directory/my_french_novel.txt")
```

This will generate three additional files in the same directory:

```
root_directory/
├── my_french_novel.txt
├── my_french_novel.tokens
├── my_french_novel.entities
└── my_french_novel.book
```

- `my_french_novel.tokens` contains all tokens along with:

  - Part-of-speech tags  
  - Syntactic parsing information  

- `my_french_novel.entities` contains information about recognized entities, including:

  - Start and end positions  
  - Entity type  

- `my_french_novel.book` contains all characters and their attributes, including:

  - Coreference information  
  - Gender, number, and other features  

## Step by Step Processing


### Step 1: Loading Models

```python
from propp_fr import load_models
spacy_model, mentions_detection_model, coreference_resolution_model = load_models()
```

Default models are:

- `spacy_model`: <a href="https://huggingface.co/spacy/fr_dep_news_trf" target="_blank">fr_dep_news_trf</a>
- `mentions_detection_model`: <a href="https://huggingface.co/AntoineBourgois/propp-fr_NER_camembert-large_FAC_GPE_LOC_PER_TIME_VEH" target="_blank">propp-fr_NER_camembert-large_FAC_GPE_LOC_PER_TIME_VEH</a>
- `coreference_resolution_model`: <a href="https://huggingface.co/AntoineBourgois/propp-fr_coreference-resolution_camembert-large_PER" target="_blank">propp-fr_coreference-resolution_camembert-large_PER</a>

### Step 2: Loading a .txt File

```python
from propp_fr import load_text_file
text_content = load_text_file("root_directory/my_french_novel.txt")
```

### Step 3: Tokenizing the Text

Break down the text into individual tokens (words and punctuation) with linguistic information:


```python
from propp_fr import generate_tokens_df
tokens_df = generate_tokens_df(text_content, spacy_model)
```

`tokens_df` is a [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) where each row represents one token from the text.

| Column Name | Description |
|------------|-------------|
| `paragraph_ID` | Which paragraph the token belongs to |
| `sentence_ID` | Which sentence the token belongs to |
| `token_ID_within_sentence` | Position of the token within its sentence |
| `token_ID_within_document` | Position of the token in the entire document |
| `word` | The actual word as it appears in the text |
| `lemma` | The base/dictionary form of the word |
| `byte_onset` | Starting byte position in the original file |
| `byte_offset` | Ending byte position in the original file |
| `POS_tag` | Part-of-speech tag (noun, verb, adjective, etc.) |
| `dependency_relation` | How the word relates to other words |
| `syntactic_head_ID` | The ID of the word this token depends on |


### Step 4: Embedding Tokens

Transform the tokens into numerical representations (embeddings) that capture their meaning:

```python
from propp_fr import load_tokenizer_and_embedding_model, get_embedding_tensor_from_tokens_df

# Load the tokenizer and pre-trained embedding model
tokenizer, embedding_model = load_tokenizer_and_embedding_model(
    mentions_detection_model["base_model_name"],
  )

# Generate embeddings for all tokens
tokens_embedding_tensor = get_embedding_tensor_from_tokens_df(
    text_content,
    tokens_df,
    tokenizer,
    embedding_model,
  )
```

`tokens_embedding_tensor` is a [`torch.tensor`](https://docs.pytorch.org/docs/stable/tensors.html) object with dimensions `[number_of_tokens, embedding_size]`.

Each row corresponds to one token from `tokens_df`, preserving the same order.

These embeddings will be used as inputs for the `mention detection model` and the `coreference resolution model`.

### Step 5: Mention Spans Detection

Identify all mentions belonging to entities of different types in the text:

- Characters (PER): pronouns (je, tu, il, ...), possessive pronouns (mon, ton, son, ...), common nouns (le capitaine, la princesse, ...) and proper nouns (Indiana Delmare, Honoré de Pardaillan, ...)  
- Facilities (FAC): chatêau, sentier, chambre, couloir, ...  
- Time (TIME): le règne de Louis XIV, ce matin, en juillet, ...  
- Geo-Political Entities (GPE): Montrouge, France, le petit hameau, ...  
- Locations (LOC): le sud, Mars, l'océan, le bois, ...  
- Vehicles (VEH): avion, voitures, calèche, vélos, ...

```python
from propp_fr import generate_entities_df

entities_df = generate_entities_df(
    tokens_df,
    tokens_embedding_tensor,
    mentions_detection_model,
)
```

**What this does:** Scans through the text to find all mentions of entities.

The `entities_df` object is a [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) where each row represents a detected mention:

| Column Name | Description |
|------------|-------------|
| `start_token` | Token ID where the mention begins |
| `end_token` | Token ID where the mention ends |
| `cat` | Type of the entity |
| `confidence` | Model's confidence score (0-1) for this detection |
| `text` | The actual text of the mention |

To learn more about how mention detection is performed under the hood, check the [Algorithms Section](algorithms/#mention_spans_detection_model)

### Step 6: Adding Linguistic Features

Enrich your entity mentions with additional grammatical and syntactic information:

```python
from propp_fr import add_features_to_entities

entities_df = add_features_to_entities(entities_df, tokens_df)
```

**What this does:** Adds detailed linguistic features to each mention, including grammatical properties, syntactic structure, and contextual information.

This step adds the following columns to `entities_df`:

| Column Name | Description |
|------------|-------------|
| `mention_len` | Length of the mention in tokens |
| `paragraph_ID` | Paragraph containing the mention |
| `sentence_ID` | Sentence containing the mention |
| `start_token_ID_within_sentence` | Position where mention starts in its sentence |
| `out_to_in_nested_level` | Nesting depth (outer to inner) |
| `in_to_out_nested_level` | Nesting depth (inner to outer) |
| `nested_entities_count` | Number of entities nested within this mention |
| `temp_index` | Temporary indexing value |
| `head_id` | ID of the syntactic head token |
| `head_word` | The actual head word |
| `head_dependency_relation` | Dependency relation of the head |
| `head_syntactic_head_ID` | ID of the head's syntactic parent |
| `POS_tag` | Part-of-speech tag of the head |
| `prop` | Mention type: pronoun (PRON), common noun (NOM), or proper noun (PROPN) |
| `number` | Grammatical number (singular/plural) |
| `gender` | Grammatical gender (masculine/feminine) |
| `grammatical_person` | Grammatical person (1st, 2nd, 3rd) |

**What you can analyze**: With these features, you can perform various literary and linguistic analyses:

- **Character centrality**: Count mention frequency to identify main characters
- **Proper name analysis**: Track occurrences of specific character or place names
- **Mention type distribution**: Analyze ratios of pronouns vs. common nouns vs. proper nouns
- **Gender representation**: Study the ratio of female vs. male character mentions
- **Narrative perspective**: Examine first-person pronoun ratios to determine narrative voice

### Step 7: Coreference Resolution

Link all mentions that refer to the same character (PER entity) (e.g., "Marie", "she", "the young woman" all referring to the same character):

```javascript
from propp_fr import perform_coreference

entities_df = perform_coreference(
    entities_df,
    tokens_embedding_tensor,
    coreference_resolution_model,
    propagate_coref=True,
    rule_based_postprocess=False
)
```

**What this does:** Groups mentions that refer to the same entity, creating coreference chains. For example, "Jean", "he", "the captain", and "his" might all be linked together as references to the same character.

**Parameters:**

- `propagate_coref`: If `True`, propagates coreference decisions through the chain  
- `rule_based_postprocess`: If `True`, applies additional rule-based refinements to the model's predictions  

To learn more about how coreference resolution is performed under the hood, check the [Algorithms Section](algorithms/#coreference-resolution-model)

### Coreference Resolution


