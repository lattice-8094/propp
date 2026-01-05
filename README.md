<h1 align="center">
  <table align="center">
    <tr>
      <td>
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="docs/images/propp_logo_favicon_white.svg">
          <img src="docs/images/propp_logo_favicon.svg" alt="PROPP logo" height="32" />
        </picture>
      </td>
      <td>
        <a href="https://lattice-8094.github.io/propp/" style="text-decoration: underline; font-weight: bold; color: blue; font-size: 1.5em;">
          PROPP DOCUMENTATION
        </a>
      </td>
      <td>
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="docs/images/propp_logo_favicon_flipped_white.svg">
          <img src="docs/images/propp_logo_favicon_flipped.svg" alt="PROPP logo mirrored" height="32" />
        </picture>
      </td>
    </tr>
  </table>
</h1>

<p align="center">
  <strong>Pattern Recognition and Ontologies for Prose Processing</strong><br />
  <em>A Natural Language Processing Framework for Narrative Analysis</em>
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/151ODFrKc4EVWojHpNoSUSvZsggGjBQ1j?usp=sharing" target="_blank" rel="noopener noreferrer">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab demo">
  </a>
  <a href="https://pypi.org/project/propp-fr/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/pypi/v/propp-fr.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/propp-fr/">
    <img src="https://img.shields.io/pypi/dm/propp-fr.svg" alt="PyPI downloads">
  </a>
  <a href="https://github.com/lattice-8094/propp/tree/main?tab=MIT-1-ov-file">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </a>
  <a href="https://lattice-8094.github.io/propp/">
    <img src="https://img.shields.io/badge/docs-available-brightgreen.svg" alt="Documentation">
  </a>
</p>

**Propp** is a modular NLP pipeline designed to extract rich character-centric information from narrative texts, especially litterature.

## Hands-on Tutorial Notebook

[`This Notebook`](https://colab.research.google.com/drive/151ODFrKc4EVWojHpNoSUSvZsggGjBQ1j) will guide you through the process of analyzing a French novel using the propp-fr library. 

You'll learn how to load a novel, tokenize it, extract named entities, resolve coreferences, and analyze the main characters.

## Quick Start

**Installation**

The French variant of the Propp python library can be installed via [pypi](https://pypi.org/project/propp-fr/):

```bash
pip install propp_fr
```

**Oneliner Processing**

You can process a text file in one line with the default models:

```python
from propp_fr import process_text_file

process_text_file("root_directory/my_french_novel.txt")
```

This will generate three additional files in the same directory:

```
root_directory/
├── my_french_novel.txt
├── my_french_novel.tokens
├── my_french_novel.entities
└── my_french_novel.book
```

## Presentation

[Presentation (in french)](https://mate-shs.cnrs.fr/actions/tutomate/tuto71_propp_bourgois/)

### Research Papers
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