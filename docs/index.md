---
title: "Propp – NLP Framework for Narrative Analysis"
---

# PROPP
## Pattern Recognition and Ontologies for Prose Processing
## A Natural Language Processing Framework for Narrative Analysis

**Propp** is a Python-based Natural Language Processing (NLP) framework for narrative analysis, character modeling, computational narratology, and ontology-based prose processing.  


## Narratology Context

## Computational Humanities

It combines deep learning and rule-based techniques for:

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

---

## Installation

```bash
pip install propp_fr
```
