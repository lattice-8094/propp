# Annotated Datasets

## NER-Only Propp formatted datasets

- [Download **conll2003 NER** dataset – PROPP Minimal Implementation](datasets/conll2003-NER_propp_minimal_implementation.zip)

**conll2003-NER** Mention Spans Detection

| Embedding Model                                        |   Micro F1 |   Macro F1 |
|:-------------------------------------------------------|-----------:|-----------:|
| answerdotai/ModernBERT-large                           |      89.81 |      88.59 |
| google-bert/bert-base-cased                            |      90.72 |      88.92 |
| google-bert/bert-large-cased                           |      91.54 |      89.89 |
| FacebookAI/xlm-roberta-large                           |      92.55 |      91    |
| google/t5-v1_1-xl                                      |      93.17 |      91.5  |
| FacebookAI/roberta-large                               |      93.17 |      91.62 |
| google/flan-t5-xl                                      |      93.54 |      92.28 |
| google/mt5-xxl                                         |      93.55 |      92.14 |
| google/mt5-xl                                          |      93.81 |      92.42 |

## Coreference Resolution Propp formatted datasets

- [Download **Long-LitBank-fr-PER-Only** dataset – PROPP Minimal Implementation](datasets/long-litbank-fr-PER-only_propp_minimal_implementation.zip)

- [Download **LitBank-fr** dataset – PROPP Minimal Implementation](datasets/litbank-fr_propp_minimal_implementation.zip)

- [Download **LitBank** dataset – PROPP Minimal Implementation](datasets/litbank_propp_minimal_implementation.zip)
> **Note:** This version is a **minimal implementation** of the [original LitBank dataset](https://github.com/dbamman/litbank), formatted specifically for use with **Propp’s coreference resolution training pipeline**. It contains only the essential columns (`byte_onset`, `byte_offset`, `cat`, `COREF_name`) aligned with the text for efficient model training.



## French Datasets

### LitBank-fr

### Long-LitBank-fr (characters only)


=== "Cite this work"
    
    Antoine Bourgois and Thierry Poibeau. 2025.
    [The Elephant in the Coreference Room: Resolving Coreference in Full-Length French Fiction Works.](https://aclanthology.org/2025.crac-1.5/)
    In *Proceedings of the Eighth Workshop on Computational Models of Reference, Anaphora and Coreference*, 
    pages 55–69, 
    Suzhou, China. 
    Association for Computational Linguistics.

=== "BibTeX"

        @inproceedings{bourgois-poibeau-2025-elephant,
            title = "The Elephant in the Coreference Room: Resolving Coreference in Full-Length {F}rench Fiction Works",
            author = "Bourgois, Antoine  and
              Poibeau, Thierry",
            editor = "Ogrodniczuk, Maciej  and
              Novak, Michal  and
              Poesio, Massimo  and
              Pradhan, Sameer  and
              Ng, Vincent",
            booktitle = "Proceedings of the Eighth Workshop on Computational Models of Reference, Anaphora and Coreference",
            month = nov,
            year = "2025",
            address = "Suzhou, China",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2025.crac-1.5/",
            doi = "10.18653/v1/2025.crac-1.5",
            pages = "55--69",
            abstract = "While coreference resolution is attracting more interest than ever from computational literature researchers, representative datasets of fully annotated long documents remain surprisingly scarce. In this paper, we introduce a new annotated corpus of three full-length French novels, totaling over 285,000 tokens. Unlike previous datasets focused on shorter texts, our corpus addresses the challenges posed by long, complex literary works, enabling evaluation of coreference models in the context of long reference chains. We present a modular coreference resolution pipeline that allows for fine-grained error analysis. We show that our approach is competitive and scales effectively to long documents. Finally, we demonstrate its usefulness to infer the gender of fictional characters, showcasing its relevance for both literary analysis and downstream NLP tasks."
        }


## English Datasets

### LitBank-en

[LitBank](https://github.com/dbamman/litbank) is an annotated dataset of 100 works of English-language fiction designed to support tasks in natural language processing and the computational humanities. 

**Note:** This version does **not modify the underlying annotations**, only restructures them for easier use in Propp.


=== "Cite this work"

    David Bamman, Olivia Lewke, and Anya Mansoor.
    2020.
    [An Annotated Dataset of Coreference in English Literature.](https://aclanthology.org/2020.lrec-1.6/) 
    In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 44–54, Marseille, France. European Language Resources Association.

=== "BibTeX"

        @inproceedings{bamman-etal-2020-annotated,
            title = "An Annotated Dataset of Coreference in {E}nglish Literature",
            author = "Bamman, David and Lewke, Olivia and Mansoor, Anya",
            booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
            year = "2020",
            address = "Marseille, France",
            publisher = "European Language Resources Association",
            url = "https://aclanthology.org/2020.lrec-1.6/",
            pages = "44--54",
            ISBN = "979-10-95546-34-4",
        }


We evaluated PROPP’s NER pipeline using multiple transformer-based embedding models. (test set 0)

| Embedding Model                                        |   Micro F1 |   Macro F1 |
|:-------------------------------------------------------|-----------:|-----------:|
| answerdotai/ModernBERT-large                           |      85.65 |      55.5  |
| google-bert/bert-base-cased                            |      87.12 |      60.96 |
| google-bert/bert-large-cased                           |      87.26 |      63.58 |
| google/mt5-xxl                                         |      87.86 |      60.89 |
| FacebookAI/xlm-roberta-large                           |      88.04 |      61.49 |
| FacebookAI/roberta-large                               |      88.08 |      61.67 |
| google/t5-v1_1-xl                                      |      88.22 |      63.26 |
| google/flan-t5-xl                                      |      88.53 |      61.95 |
| google/mt5-xl                                          |      89.1  |      68.74 |

## Russian Datasets

🚧 Coming soon... 🚧