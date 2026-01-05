# Datasets Benchmarks

## NER datasets

**conll2003-NER** Mention Spans Detection ([test set](https://aclanthology.org/W03-0419/))

| Embedding Model                                        |   Micro F1 |   Macro F1 | Support   |
|:-------------------------------------------------------|-----------:|-----------:|:----------|
| answerdotai/ModernBERT-large                           |      89.81 |      88.59 | 5,648     |
| google-bert/bert-base-cased                            |      90.72 |      88.92 | 5,648     |
| google-bert/bert-large-cased                           |      91.54 |      89.89 | 5,648     |
| FacebookAI/xlm-roberta-large                           |      92.55 |      91.00 | 5,648     |
| google/t5-v1_1-xl                                      |      93.17 |      91.50 | 5,648     |
| FacebookAI/roberta-large                               |      93.17 |      91.62 | 5,648     |
| google/flan-t5-xl                                      |      93.54 |      92.28 | 5,648     |
| google/mt5-xxl                                         |      93.55 |      92.14 | 5,648     |
| google/mt5-xl                                          |      93.81 |      92.42 | 5,648     |


**OntoNotes 5 - NER** Mention Spans Detection ([test set](https://cemantix.org/data/ontonotes.html))

| Embedding Model              |   Micro F1 |   Macro F1 | Support   |
|:-----------------------------|-----------:|-----------:|:----------|
| answerdotai/ModernBERT-large |      88.11 |      77.26 | 11,257    |
| google-bert/bert-base-cased  |      89.13 |      80.03 | 11,257    |
| google-bert/bert-large-cased |      89.31 |      79.11 | 11,257    |
| google/flan-t5-xl            |      89.67 |      79.57 | 11,257    |
| FacebookAI/roberta-large     |      90.55 |      81.96 | 11,257    |
| google/mt5-xl                |      90.58 |      82.35 | 11,257    |
| FacebookAI/xlm-roberta-large |      90.60 |      81.63 | 11,257    |


### LitBank-fr

### Long-LitBank-fr (characters only)

We evaluated PROPP’s NER pipeline using multiple transformer-based embedding models. (LOOCV)
`PER` mentions only

| Embedding Model                        |   Micro F1 |   Macro F1 | Support   |
|:---------------------------------------|-----------:|-----------:|:----------|
| almanach/moderncamembert-cv2-base      |      92.08 |      92.08 | 71,883    |
| dbmdz/bert-base-french-europeana-cased |      93.95 |      93.95 | 71,883    |
| FacebookAI/xlm-roberta-large           |      94.40 |      94.40 | 71,883    |
| almanach/camembert-base                |      94.53 |      94.53 | 71,883    |
| google/mt5-xl                          |      94.56 |      94.56 | 71,883    |
| almanach/camembert-large               |      94.68 |      94.68 | 71,883    |


### LitBank-en

We evaluated PROPP’s NER pipeline using multiple transformer-based embedding models. ([test set 0](https://github.com/dbamman/lrec2020-coref/tree/master/data/litbank_tenfold_splits/0))

| Embedding Model                                        |   Micro F1 |   Macro F1 | Support   |
|:-------------------------------------------------------|-----------:|-----------:|:----------|
| answerdotai/ModernBERT-large                           |      85.65 |      55.50 | 2,832     |
| google-bert/bert-base-cased                            |      87.12 |      60.96 | 2,832     |
| google-bert/bert-large-cased                           |      87.26 |      63.58 | 2,832     |
| google/mt5-xxl                                         |      87.86 |      60.89 | 2,832     |
| FacebookAI/xlm-roberta-large                           |      88.04 |      61.49 | 2,832     |
| FacebookAI/roberta-large                               |      88.08 |      61.67 | 2,832     |
| google/t5-v1_1-xl                                      |      88.22 |      63.26 | 2,832     |
| google/flan-t5-xl                                      |      88.53 |      61.95 | 2,832     |
| google/mt5-xl                                          |      89.10 |      68.74 | 2,832     |

## Russian Datasets

🚧 Coming soon... 🚧