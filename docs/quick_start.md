---
title: "Quick Start"
---

# Quick Start

## Google Colab Hands-on Tutorial

[`This Notebook`](https://colab.research.google.com/drive/151ODFrKc4EVWojHpNoSUSvZsggGjBQ1j) will guide you through the process of analyzing a French novel using the propp-fr library. 

You'll learn how to load a novel, tokenize it, extract named entities, resolve coreferences, and analyze the main characters.

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

## Reloading Processed Files

Generated files are loaded by:

```python
from propp_fr import load_text_file, load_tokens_df, load_entities_df, load_book_file

file_name = "my_french_novel"
root_directory = "root_directory"

text_content = load_text_file(file_name, root_directory)
tokens_df = load_tokens_df(file_name, root_directory)
entities_df = load_entities_df(file_name, root_directory)
characters_dict = load_book_file(file_name, root_directory)
```