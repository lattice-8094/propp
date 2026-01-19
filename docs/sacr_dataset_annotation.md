# Annotating a new dataset

## Why Annotating a New Dataset ?

Sometimes the entity types you are interested in are not covered by the [Available Dataset](available_datasets/), or the language you are working with is not supported. 
In these cases, you will need to create a dataset from scratch.

This page will guide you through the process of annotating a new dataset, from preparing your corpus to generating the annotation files used to train a new model with Propp.

## Creating a new dataset from scratch.

### Collecting texts for the corpus

To create a new dataset, you must first build the corpus to be annotated. The corpus should consist of one or more plain `.txt` files.

Text selection should reflect the intended application of the model. 
There is a trade-off between corpus size and annotation effort: more data improves coverage but requires more time to annotate. 
In practice, a small but diverse corpus is often preferable to a large, homogeneous one (e.g. varying authors, genres, and periods).

If the dataset is intended for public release, the source texts must be in the public domain.

Copyright duration depends on jurisdiction; see: https://en.wikipedia.org/wiki/List_of_copyright_duration_by_country

Suitable sources include:

  - [Wikisource](https://en.wikisource.org/wiki/Main_Page)
  - [Project Gutenberg](https://gutenberg.org/)
  - [Standard Ebooks](https://standardebooks.org)
  - [Gallica](https://gallica.bnf.fr/accueil/en/html/accueil-en)

#### Example

Suppose we want to train a model that detects all mentions referring to `animal entities`. 

To construct a minimal corpus, we can start from Wikipedia pages related to animals.

!!! note
    Wikipedia content is released under the [**CC BY-SA** license](https://creativecommons.org/licenses/by-sa/4.0/deed.en) and is **not in the public domain**.  
   

In this example, we construct the corpus from Wikipedia pages corresponding to a small set of animal species (Dog, Guinea pig, Cow, Lion, Jellyfish, Flamingo, Kangaroo).

??? Abstract "Python Code to Generate the .txt Animals Corpus"

    ```python
    import wikipedia, os
    
    def get_wikipedia_text_from_title(url: str) -> str:
        # Extract the page title from the URL
        title = url.split("/wiki/")[-1]
        title = title.replace("_", " ")
    
        # Fetch the page content
        page = wikipedia.page(title, auto_suggest=False)
        return page.content
    
    corpus_root_path = "./animals_corpus"
    os.makedirs(corpus_root_path, exist_ok=True)
    
    animals = ["Dog", "Guinea pig", "Cow", "Lion", "Jellyfish", "Flamingo", "Kangaroo"]
    
    for animal in animals:
        title = animal.replace(" ", "_").lower()
        filename = title + ".txt"
        filepath = os.path.join(corpus_root_path, filename)
        text = get_wikipedia_text_from_title(animal)
    
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    ```

The resulting corpus has the following structure:

```
animals_corpus/
├── cow.txt
├── dog.txt
├── flamingo.txt
├── guinea_pig.txt
├── jellyfish.txt
├── kangaroo.txt
└── lion.txt
```

This corpus serves as the foundation for the annotation process. 

Before starting the actual annotation, it is essential to clearly define what entities should be annotated and how mentions should be identified. 

This ensures consistency and reproducibility across the dataset.

### Annotation Guidelines

Annotation guidelines are structured instructions that define how a dataset should be annotated.

They serve as a reference for annotators to ensure that entities are identified consistently and reproducibly across the corpus. 

Well-defined guidelines are essential for dataset quality, inter-annotator agreement, and for training models that generalize effectively.


A good annotation guideline should include:

  1. Precise definitions of the entities or phenomena to annotate.
    - Clearly distinguish between closely related entity types. For example, in a narrative corpus, “Animal” vs. “Mythical Creature” should have unambiguous criteria.
  - Examples of positive and negative cases.
    - Positive: “The `lion` roared.” → `lion` is an Animal.
    - Negative: “The Panthera leo genus is diverse.” → “Panthera leo” may or may not be annotated depending on your scope.
  - Annotation rules for edge cases.
    - Pronouns referring to annotated entities.
    - Proper nouns vs. common nouns.
    - Mentions in foreign or Latin names.
  - Consistency instructions for dealing with ambiguous or borderline cases.
    - For example, always prefer the more specific type when a mention could belong to two categories.

Example (Animal annotation):

Annotate all mentions of specific animal species: “Dog”, “Lion”, “Jellyfish”

Do not annotate higher-level taxa: “Canine”, “Felidae”

Annotate pronouns referring to these entities: “it”, “they”

Annotate Latin species names if they appear in context: Panthera leo

For more detailed examples and templates of well-structured guidelines, see:

### Annotation guidelines examples

#### Litbank

https://github.com/dbamman/litbank?tab=readme-ov-file

#### Litbank-fr

https://github.com/lattice-8094/Proppagate/blob/main/fr/data/Manuel_Annotation_propp.md

Transition vers annotation process

### Annotation Process

Multiple annotation tools exist. For Propp we use [SACR](https://boberle.com/projects/coreference-annotation-with-sacr/)

=== "Cite this work"
    
    Bruno Oberle. 2018. 
    [SACR: A Drag-and-Drop Based Tool for Coreference Annotation.](https://aclanthology.org/L18-1059/)
    In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan.
    European Language Resources Association (ELRA).

=== "BibTeX"

        @InProceedings{OBERLE18.178,  
          author = {Bruno Oberle},
          title = "{SACR: A Drag-and-Drop Based Tool for Coreference Annotation}",
          booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
          year = {2018},
          month = {May 7-12, 2018},
          address = {Miyazaki, Japan},
          editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and Hélène Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
          publisher = {European Language Resources Association (ELRA)},
          isbn = {979-10-95546-00-9},
          language = {english}
          }

Before starting the annotation in SACR, you need to define the properties configuration : 

Here we only annotate "ANIMAL" entities, but if we wanted to also annotate geopolitical entities (GPE) we would add a new row, one row per entity.

```
PROP:name=EN
$$$
ANIMAL
GPE
```

Example of other annotation properties: 

litbank-fr: 7 ['FAC', 'TIME', 'GPE', 'ORG', 'VEH', 'LOC', 'PER']  
litbank-ru : 6 ['FAC', 'GPE', 'ORG', 'VEH', 'LOC', 'PER']  
litbank : 6 ['FAC', 'GPE', 'ORG', 'VEH', 'LOC', 'PER']  

1. Open the [SACR annotation Home Page](https://boberle.com/projects/coreference-annotation-with-sacr/online/)
2. Load your raw .txt file
3. Paste you Properties configuration
4. Choose the tokenization type: `word and punctuation`
5. Then click the button to `parse the data`
6. Annotate your entities mention spans boundaries 
7. Select entity type 
8. Link coreferential mentions 
9. Name your coreference chain (select a mention and press `n`)
10. Save your annotated file (press `w`) **Save regularly to avoid losing annotations**

<video id="sacr-video" autoplay muted loop playsinline controls style="max-width: 100%; height: auto;">
  <source src="../images/SACR_annotations.webm" type="video/webm">
  Your browser does not support the video tag.
</video>


