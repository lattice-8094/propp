# Coreference Annotated Datasets

## Propp formatted datasets

- [Download **Long-LitBank-fr-PER-Only** dataset – PROPP Minimal Implementation](datasets/long-litbank-fr-PER-only_propp_minimal_implementation.zip)

- [Download **LitBank-fr** dataset – PROPP Minimal Implementation](datasets/litbank-fr_propp_minimal_implementation.zip)

- [Download **LitBank** dataset – PROPP Minimal Implementation](datasets/litbank_propp_minimal_implementation.zip)
> **Note:** This version is a **minimal implementation** of the [original LitBank dataset](https://github.com/dbamman/litbank), formatted specifically for use with **Propp’s coreference resolution training pipeline**. It contains only the essential columns (`byte_onset`, `byte_offset`, `cat`, `COREF_name`) aligned with the text for efficient model training.



## French Datasets

### LitBank-fr

### Long-LitBank-fr (characters only)

## English Datasets

### LitBank-en

[LitBank](https://github.com/dbamman/litbank) is an annotated dataset of 100 works of English-language fiction designed to support tasks in natural language processing and the computational humanities. 
The dataset is described in the following publications:

- David Bamman, Sejal Popat, and Sheng Shen (2019), "[An Annotated Dataset of Literary Entities](http://people.ischool.berkeley.edu/~dbamman/pubs/pdf/naacl2019_literary_entities.pdf)," *NAACL 2019*.
- Matthew Sims, Jong Ho Park, and David Bamman (2019), "[Literary Event Detection](http://people.ischool.berkeley.edu/~dbamman/pubs/pdf/acl2019_literary_events.pdf)," *ACL 2019*.
- David Bamman, Olivia Lewke, and Anya Mansoor (2020), "[An Annotated Dataset of Coreference in English Literature](https://arxiv.org/abs/1912.01140)," *LREC*.

**Note:** This version does **not modify the underlying annotations**, only restructures them for easier use in Propp.

```
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
```

## Russian Datasets

🚧 Coming soon... 🚧