# Datasets Benchmarks

## NER datasets

/data-lachesis/common/propp/.venv/lib/python3.11/site-packages/propp_fr/propp_fr_add_entities_features.py:4: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
propp_fr package loaded successfully.


### conll2003-NER


Test Splits: 1 [231 File(s) / split]  |  Overall Tested Ratio: 16.58% [231/1393 Files]

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

Best Embedding Model: google/mt5-xl

| NER_tag   |   precision |   recall |   f1_score | support   |
|:----------|------------:|---------:|-----------:|:----------|
| LOC       |       94.93 |    94.30 |      94.62 | 1,668     |
| ORG       |       91.72 |    93.32 |      92.51 | 1,661     |
| PER       |       98.09 |    98.33 |      98.21 | 1,617     |
| MISC      |       83.71 |    84.19 |      83.95 | 702       |
| micro_avg |       93.49 |    93.91 |      93.70 | 5,648     |
| macro_avg |       92.11 |    92.54 |      92.32 | 5,648     |

================================================================================


### litbank


Test Splits: 10 [10 File(s) / split]  |  Overall Tested Ratio: 100.00% [100/100 Files]

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

Best Embedding Model: google/mt5-xl

| NER_tag   |   precision |   recall |   f1_score | support   |
|:----------|------------:|---------:|-----------:|:----------|
| PER       |       93.56 |    93.55 |      93.55 | 24,180    |
| FAC       |       68.70 |    65.67 |      67.15 | 2,330     |
| LOC       |       66.84 |    61.13 |      63.86 | 1,289     |
| GPE       |       78.60 |    72.05 |      75.18 | 948       |
| VEH       |       73.48 |    64.25 |      68.56 | 207       |
| ORG       |       55.56 |    16.78 |      25.77 | 149       |
| micro_avg |       89.56 |    88.58 |      89.00 | 29,103    |
| macro_avg |       72.79 |    62.24 |      65.68 | 29,103    |

================================================================================


### litbank-fr


Test Splits: 29 [1 File(s) / split]  |  Overall Tested Ratio: 100.00% [29/29 Files]

| Embedding Model                   |   Micro F1 |   Macro F1 | Support   |
|:----------------------------------|-----------:|-----------:|:----------|
| almanach/moderncamembert-cv2-base |      84.43 |      54.84 | 38,630    |
| almanach/moderncamembert-base     |      85.31 |      57.20 | 38,630    |
| FacebookAI/xlm-roberta-large      |      87.70 |      60.39 | 38,630    |
| almanach/camembert-base           |      87.76 |      61.43 | 38,630    |
| google/mt5-xl                     |      88.16 |      61.62 | 38,630    |
| almanach/camembert-large          |      88.19 |      62.43 | 38,630    |

Best Embedding Model: almanach/camembert-large

| NER_tag   |   precision |   recall |   f1_score | support   |
|:----------|------------:|---------:|-----------:|:----------|
| PER       |       92.22 |    93.68 |      92.94 | 32,349    |
| FAC       |       69.26 |    72.01 |      70.61 | 2,297     |
| TIME      |       58.45 |    59.18 |      58.81 | 1,683     |
| GPE       |       76.00 |    76.61 |      76.31 | 868       |
| LOC       |       62.18 |    46.73 |      53.36 | 781       |
| VEH       |       65.49 |    47.95 |      55.36 | 463       |
| ORG       |       40.74 |    23.28 |      29.63 | 189       |
| micro_avg |       87.84 |    88.66 |      88.19 | 38,630    |
| macro_avg |       66.33 |    59.92 |      62.43 | 38,630    |

================================================================================


### long-litbank-fr-PER-only


Test Splits: 32 [1 File(s) / split]  |  Overall Tested Ratio: 100.00% [32/32 Files]

| Embedding Model                   |   Micro F1 |   Macro F1 | Support   |
|:----------------------------------|-----------:|-----------:|:----------|
| almanach/moderncamembert-cv2-base |      91.94 |      91.94 | 71,883    |
| almanach/moderncamembert-base     |      92.74 |      92.74 | 71,883    |
| FacebookAI/xlm-roberta-large      |      94.45 |      94.45 | 71,883    |
| almanach/camembert-base           |      94.57 |      94.57 | 71,883    |
| almanach/camembert-large          |      94.74 |      94.74 | 71,883    |
| google/mt5-xl                     |      94.74 |      94.74 | 71,883    |

Best Embedding Model: google/mt5-xl

| NER_tag   |   precision |   recall |   f1_score | support   |
|:----------|------------:|---------:|-----------:|:----------|
| PER       |       94.96 |    94.53 |      94.74 | 71,883    |
| micro_avg |       94.96 |    94.53 |      94.74 | 71,883    |
| macro_avg |       94.96 |    94.53 |      94.74 | 71,883    |

================================================================================


### ontonotes5_english-NER


Test Splits: 1 [207 File(s) / split]  |  Overall Tested Ratio: 5.69% [207/3637 Files]

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

Best Embedding Model: FacebookAI/roberta-large

| NER_tag     |   precision |   recall |   f1_score | support   |
|:------------|------------:|---------:|-----------:|:----------|
| GPE         |       97.75 |    97.01 |      97.38 | 2,240     |
| PERSON      |       96.45 |    95.62 |      96.03 | 1,988     |
| ORG         |       91.82 |    92.59 |      92.21 | 1,795     |
| DATE        |       87.25 |    89.21 |      88.22 | 1,603     |
| CARDINAL    |       85.15 |    81.05 |      83.05 | 934       |
| NORP        |       94.43 |    94.77 |      94.60 | 841       |
| PERCENT     |       89.46 |    89.97 |      89.71 | 349       |
| MONEY       |       87.42 |    88.54 |      87.97 | 314       |
| TIME        |       66.67 |    63.21 |      64.89 | 212       |
| ORDINAL     |       85.00 |    87.18 |      86.08 | 195       |
| LOC         |       75.00 |    80.45 |      77.63 | 179       |
| WORK_OF_ART |       78.43 |    72.29 |      75.24 | 166       |
| FAC         |       80.67 |    71.11 |      75.59 | 135       |
| QUANTITY    |       80.18 |    84.76 |      82.41 | 105       |
| PRODUCT     |       75.90 |    82.89 |      79.25 | 76        |
| EVENT       |       71.23 |    82.54 |      76.47 | 63        |
| LAW         |       70.97 |    55.00 |      61.97 | 40        |
| LANGUAGE    |       85.71 |    54.55 |      66.67 | 22        |
| micro_avg   |       91.01 |    90.73 |      90.84 | 11,257    |
| macro_avg   |       83.31 |    81.26 |      81.96 | 11,257    |

================================================================================


**conll2003-NER** Mention Spans Detection ([test set](https://aclanthology.org/W03-0419/))

**OntoNotes 5 - NER** Mention Spans Detection ([test set](https://cemantix.org/data/ontonotes.html))

**litbank-en** We evaluated PROPP’s NER pipeline using multiple transformer-based embedding models. ([test sets](https://github.com/dbamman/lrec2020-coref/tree/master/data/litbank_tenfold_splits/))

## Coreference Resolution datasets


### litbank (GOLD mentions)

Test Splits: 1 [10 File(s) / split]  |  Overall Tested Ratio: 10.00% [10/100 Files]

| embedding_model              | avg_tokens   |   MUC_f1 |   B3_f1 |   CEAFe_f1 |   CONLL_f1 |
|:-----------------------------|:-------------|---------:|--------:|-----------:|-----------:|
| answerdotai/ModernBERT-large | 2,091        |    86.07 |   70.49 |      74.47 |      77.01 |
| FacebookAI/xlm-roberta-large | 2,091        |    87.12 |   74.06 |      76.15 |      79.11 |
| google-bert/bert-large-cased | 2,091        |    87.92 |   73.26 |      77.20 |      79.46 |
| google/mt5-xl                | 2,091        |    88.73 |   77.53 |      77.23 |      81.17 |
| FacebookAI/roberta-large     | 2,091        |    89.03 |   78.31 |      77.96 |      81.77 |
| google/flan-t5-xl            | 2,091        |    89.25 |   78.66 |      78.76 |      82.22 |


### litbank-fr (GOLD mentions)

Test Splits: 29 [1 File(s) / split]  |  Overall Tested Ratio: 100.00% [29/29 Files]

| embedding_model                   | avg_tokens   |   MUC_f1 |   B3_f1 |   CEAFe_f1 |   CONLL_f1 |
|:----------------------------------|:-------------|---------:|--------:|-----------:|-----------:|
| almanach/moderncamembert-cv2-base | 9,551        |    85.62 |   56.20 |      64.64 |      68.82 |
| almanach/moderncamembert-base     | 9,551        |    87.12 |   59.31 |      67.00 |      71.14 |
| almanach/camembert-base           | 9,551        |    87.95 |   63.31 |      65.66 |      72.31 |
| FacebookAI/xlm-roberta-large      | 9,551        |    90.05 |   68.46 |      69.43 |      75.98 |
| almanach/camembert-large          | 9,551        |    90.60 |   69.79 |      71.29 |      77.23 |


### long-litbank-fr-PER-only (GOLD mentions)

Test Splits: 32 [1 File(s) / split]  |  Overall Tested Ratio: 100.00% [32/32 Files]

| embedding_model                   | avg_tokens   |   MUC_f1 |   B3_f1 |   CEAFe_f1 |   CONLL_f1 |
|:----------------------------------|:-------------|---------:|--------:|-----------:|-----------:|
| almanach/moderncamembert-cv2-base | 17,378       |    85.07 |   45.69 |      40.78 |      57.18 |
| almanach/moderncamembert-base     | 17,378       |    85.38 |   47.17 |      42.46 |      58.34 |
| almanach/camembert-base           | 17,378       |    89.05 |   54.92 |      46.14 |      63.37 |
| FacebookAI/xlm-roberta-large      | 17,378       |    91.29 |   62.87 |      52.59 |      68.91 |
| almanach/camembert-large          | 17,378       |    91.99 |   65.07 |      56.52 |      71.19 |
| google/mt5-xl                     | 17,378       |    92.86 |   69.54 |      57.58 |      73.32 |