<h1 align="center">
  <img
    src="docs/images/propp_logo_favicon.svg"
    alt="PROPP logo"
    height="32"
    style="vertical-align: middle;"
  />
  <a href="https://lattice-8094.github.io/propp/">PROPP DOCUMENTATION</a>
  <img
    src="docs/images/propp_logo_favicon_flipped.svg"
    alt="PROPP logo mirrored"
    height="32"
    style="vertical-align: middle; transform: scaleX(-1);"
  />
</h1>






![Model Architecture](propp_logo.png)

Pattern Recognition and Ontologies for Prose Processing

Natural Language Processing Pipeline for French Literary Works

## Overview

**Propp** is a modular NLP pipeline designed to extract rich character-centric information from narrative texts, especially litterature. It combines deep learning and rule-based techniques for:

### 1. Token Embeddings  
Contextualized embeddings from pretrained transformer-based models (e.g., `google/mt5-xl`, `almanach/camembert-large`).

### 2. Mention-Spans Detection  
Identifying spans in the text that refer to named or nominal entities, as well as referential pronouns.

### 3. Mention Type Prediction  
Classifying detected mentions as persons, locations, organizations, geopolitical entities.

### 4. Coreference Resolution  
Linking mentions that refer to the same underlying entity across the narrative.

### 5. Characters Detection  
Consolidating coreferent mentions into coherent character entities.

### 6. Character-Related Attributes Extraction  
Inferring gender, social role, narrative function, and other descriptors from context.

### 7. Character Representation  
Producing structured profiles for characters (e.g., via embeddings), ready for downstream analysis or visualization.

## Python Library

pip install propp_fr
[Python Library](https://pypi.org/project/propp-fr/)

## Hands-on Tutorial Notebook

[Hands-on Tutorial Notebook](https://colab.research.google.com/drive/151ODFrKc4EVWojHpNoSUSvZsggGjBQ1j?usp=sharing)

## Presentation

[Presentation (in french)](https://mate-shs.cnrs.fr/actions/tutomate/tuto71_propp_bourgois/)

### Papers
Antoine Bourgois and Thierry Poibeau.
2025.
[The Elephant in the Coreference Room: Resolving Coreference in Full-Length French Fiction Works.](https://arxiv.org/pdf/2510.15594)
In *Proceeding of the Eighth Workshop on Computational Models of Reference, Anaphora and Coreference (CRAC 2025).* EMNLP 2025, Suzhou, China.
[arxiv](https://arxiv.org/abs/2510.15594), [hal](https://hal.science/hal-05319970).

Jean Barré, Olga Seminck, Antoine Bourgois, Thierry Poibeau.
2025.
[Modeling the Construction of a Literary Archetype: The Case of the Detective Figure in French Literature](https://arxiv.org/pdf/2511.00627)
In *Proceeding of the Sixth Conference on Computational Humanities Research 2025 (CHR 2025).* Luxembourg, Luxembourg.
[arxiv](https://arxiv.org/abs/2511.00627).
