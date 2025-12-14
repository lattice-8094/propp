---
title: "API"
---

# API Reference

## Core Functions

### load_models()
**Description**: Loads the three pre-trained models needed for analysis  
**Parameters**: None  
**Returns**:

  - spacy_model: SpaCy French model  
  - mentions_detection_model: Entity detection model  
  - coreference_resolution_model: Coreference model

**Example**: See Quick Start  
**Time**: ~30 seconds first run (downloads models)  

### load_text_file(file_path)
**Description**: Loads a French text file  
**Parameters**:

  - file_path (str): Path to .txt file

**Returns**: str - Text content  
**Example**: `text = load_text_file("novel.txt")`  