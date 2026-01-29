---
title: "Generate Character Network"
description: "Tutorial to generate character network using PROPP pipeline."
---

# Processing all .txt Files in a Directory

Use this code to generate character network from Propp output files.

??? Abstract "**You can copy / paste the whole Notebook Code**"

    ```python
    from propp_fr import generate_character_network    

    file_name = "/home/antoine/Bureau/1872_Verne-Jules_Le-tour-du-monde-en-quatre-vingts-jours/1872_Verne-Jules_Le-tour-du-monde-en-quatre-vingts-jours"

    network_metrics_df, G, fig = generate_character_network(
        file_name = file_name,
        top_n =10,
        keep_only_singular=True,
        save_outputs = True,
    )
    ```
