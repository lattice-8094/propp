# Datasets Benchmarks

## NER datasets

### conll2003-NER

| Embedding Model              |   Micro F1 |   Macro F1 | Support   |
|:-----------------------------|-----------:|-----------:|:----------|
| answerdotai/ModernBERT-large |      89.59 |      88.07 | 5,648     |
| google-bert/bert-base-cased  |      90.95 |      89.28 | 5,648     |
| google-bert/bert-large-cased |      91.51 |      89.81 | 5,648     |
| FacebookAI/xlm-roberta-large |      92.75 |      91.26 | 5,648     |
| FacebookAI/roberta-large     |      93.05 |      91.45 | 5,648     |
| google/t5-v1_1-xl            |      93.39 |      91.85 | 5,648     |
| google/flan-t5-xl            |      93.57 |      92.38 | 5,648     |
| google/mt5-xl                |      93.70 |      92.32 | 5,648     |


### litbank

| Embedding Model              |   Micro F1 |   Macro F1 | Support   |
|:-----------------------------|-----------:|-----------:|:----------|
| answerdotai/ModernBERT-large |      86.09 |      58.90 | 29,103    |
| google-bert/bert-base-cased  |      87.62 |      62.08 | 29,103    |
| google-bert/bert-large-cased |      87.93 |      64.56 | 29,103    |
| google/t5-v1_1-xl            |      88.65 |      66.27 | 29,103    |
| FacebookAI/xlm-roberta-large |      88.70 |      66.69 | 29,103    |
| FacebookAI/roberta-large     |      88.85 |      67.02 | 29,103    |
| google/flan-t5-xl            |      88.96 |      66.24 | 29,103    |
| google/mt5-xl                |      89.00 |      65.68 | 29,103    |


### litbank-fr

| Embedding Model              |   Micro F1 |   Macro F1 | Support   |
|:-----------------------------|-----------:|-----------:|:----------|
| FacebookAI/xlm-roberta-large |      86.59 |      57.90 | 19,801    |
| almanach/camembert-large     |      86.99 |      61.02 | 25,395    |
| almanach/camembert-base      |      87.76 |      61.43 | 38,630    |


### long-litbank-fr-PER-only

| Embedding Model         |   Micro F1 |   Macro F1 | Support   |
|:------------------------|-----------:|-----------:|:----------|
| almanach/camembert-base |      95.30 |      95.30 | 31,166    |


### ontonotes5_english-NER

| Embedding Model              |   Micro F1 |   Macro F1 | Support   |
|:-----------------------------|-----------:|-----------:|:----------|
| answerdotai/ModernBERT-large |      88.66 |      78.53 | 11,257    |
| google-bert/bert-large-cased |      89.40 |      79.53 | 11,257    |
| google-bert/bert-base-cased  |      89.42 |      80.20 | 11,257    |
| google/flan-t5-xl            |      90.52 |      81.81 | 11,257    |
| FacebookAI/xlm-roberta-large |      90.59 |      81.32 | 11,257    |
| google/mt5-xl                |      90.66 |      82.30 | 11,257    |
| google/t5-v1_1-xl            |      90.74 |      82.05 | 11,257    |
| FacebookAI/roberta-large     |      90.84 |      81.96 | 11,257    |




=============

**conll2003-NER** Mention Spans Detection ([test set](https://aclanthology.org/W03-0419/))

| Embedding Model              |   Micro F1 |   Macro F1 | Support   |
|:-----------------------------|-----------:|-----------:|:----------|
| answerdotai/ModernBERT-large |      89.59 |      88.07 | 5,648     |
| google-bert/bert-base-cased  |      90.95 |      89.28 | 5,648     |
| google-bert/bert-large-cased |      91.51 |      89.81 | 5,648     |
| FacebookAI/xlm-roberta-large |      92.75 |      91.26 | 5,648     |
| FacebookAI/roberta-large     |      93.05 |      91.45 | 5,648     |
| google/t5-v1_1-xl            |      93.39 |      91.85 | 5,648     |
| google/mt5-xxl               |      93.54 |      92.22 | 5,648     |
| google/flan-t5-xl            |      93.57 |      92.38 | 5,648     |
| google/mt5-xl                |      93.70 |      92.32 | 5,648     |


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


## Coreference Resolution datasets

**litbank_propp_minimal_implementation** Coreference Resolution (Gold mentions)

| Embedding Model              |   MUC |    B3 |   CEAFe |   CONLL |   Avg. Tokens Count |   Avg. Mentions Count |   Doc. |
|:-----------------------------|------:|------:|--------:|--------:|--------------------:|----------------------:|-------:|
| FacebookAI/xlm-roberta-large | 88.05 | 77.00 |   76.67 |   80.57 |                2136 |                   291 |    100 |
| google-bert/bert-base-cased  | 86.00 | 72.93 |   74.03 |   77.66 |                2136 |                   291 |    100 |
| google-bert/bert-large-cased | 86.53 | 74.94 |   75.08 |   78.85 |                2136 |                   291 |    100 |
| google/mt5-xl                | 89.63 | 80.61 |   78.76 |   83.00 |                2136 |                   291 |    100 |

