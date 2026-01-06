---
title: "About"
---

**Propp** is a **Python-based Natural Language Processing (NLP) framework** developed for the analysis of **narrative texts**, with a focus on **long-form literary fiction**. 

Designed for **computational literary studies**, Propp bridges **Natural Language Processing** and **narrative theory** to enable **character-centric**, **ontology-based**, and **corpus-scale analysis** of prose narratives.

The framework is designed for research on characters, narrative structure, and large-scale literary corpora, while remaining accessible to scholars with limited programming experience.

## Objectives

Bridging narratological theory and contemporary research in natural language processing.

## Narratology Context

Vladimir Propp

## Computational Humanities


## Design Philosophy

  - Modularity
  - Open Source
  - Informed by narratology theory

## French Fiction and Beyond

While the starting point of the project is long form French fiction narratives (novels) we aim at exploring other types of nnarratives.

Dream narratives, altered state of consciousness narratives, personal narratives, myths and legends.

## Project Lineage

  - 2014 - [Java BookNLP](https://github.com/dbamman/book-nlp?tab=readme-ov-file)
    
    - [A Bayesian Mixed Effects Model of Literary Character](https://aclanthology.org/P14-1035/) (Bamman et al., ACL 2014)
  
  - 2019 - [BookNLP](https://github.com/booknlp/booknlp)
    
    - [An annotated dataset of literary entities](https://aclanthology.org/N19-1220/) (Bamman et al., NAACL 2019)

  - 2020 - [Multilingual BookNLP](https://www.neh.gov/sites/default/files/inline-files/FOIA%2021-09%20Regents%20of%20the%20University%20of%20California%2C%20Berkeley.pdf)
  - 2021 - [BookNLP-fr](https://www.lattice.cnrs.fr/projets/booknlp/)
    
    - [BookNLP-fr, the French Versant of BookNLP. A Tailored Pipeline for 19th and 20th Century French Literature (Mélanie et al., JCLS 2024)](https://jcls.io/article/id/3924/)
  
  - 2025 - [Propp](#)

## Modules

Propp framework combines deep learning and rule-based techniques for:

  1. **Token Embeddings**

    Contextualized embeddings from pretrained transformer-based models (e.g., `google/mt5-xl`, `almanach/camembert-large`).

  2. **Mention-Spans Detection**

    Identifying spans in the text that refer to named or nominal entities, as well as referential pronouns.

  3. **Mention Type Prediction**

    Classifying detected mentions as persons, locations, organizations, geopolitical entities.

  4. **Coreference Resolution**

    Linking mentions that refer to the same underlying entity across the narrative.

  5. **Characters Detection**

    Consolidating coreferent mentions into coherent character entities.

  6. **Character-Related Attributes Extraction** 

    Inferring gender, social role, narrative function, and other descriptors from context.

  7. **Character Representation**

    Producing structured profiles for characters (e.g., via embeddings), ready for downstream analysis or visualization.

---

## Funding

This project was funded in part by the
[PRAIRIE–PSAI](https://www.prairie-psai.fr/)
(Paris Artificial Intelligence Research Institute – Paris School of Artificial Intelligence),
reference
[ANR-22-IACL-0008](https://anr.fr/ProjetIA-23-IACL-0008).

The project also received support from the Major Research Program of PSL Research University,
["CultureLab"](https://www.culturelab.psl.eu/),
launched by PSL Research University and implemented by the French National Research Agency (ANR),
reference
[ANR-10-IDEX-0001](https://anr.fr/ProjetIA-10-IDEX-0001).

The project additionally benefited from the support of the [*Fondation de l’ENS*](https://fondation.ens.psl.eu/en/accueil-english/), through the *Be Ys Chair*, dedicated to the support of innovative research projects in
artificial intelligence.