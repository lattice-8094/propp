---
title: "API"
---

# API

## propp_fr.load_text_file

```python
propp_fr.load_text_file(file_name: str, files_directory: str = "", extension: str = ".txt") → str
```

Loads and returns the content of a text file from a specified directory.

This function automatically appends the file extension if not already present in the filename, constructs the full file path using `pathlib.Path`, and reads the entire content of the file using UTF-8 encoding.

??? note "propp_fr.load_text_file"

    **Parameters**
    
    - **file_name** (*str*) – The name of the file to load. If the extension is not included, it will be automatically appended.
    - **files_directory** (*str, optional*) – The directory path where the file is located. If empty, looks in the current working directory. Default: `""`.
    - **extension** (*str, optional*) – The file extension to append if not present in `file_name`. Default: `".txt"`.
    
    ### Returns
    
    **str** – The complete text content of the file as a string.
    
    ### Raises
    
    - **FileNotFoundError** – If the specified file does not exist at the given path.
    - **PermissionError** – If the file cannot be read due to insufficient permissions.
    - **UnicodeDecodeError** – If the file cannot be decoded using UTF-8 encoding.
    
    ### Examples
    
    ```python
    # Load a file from the current directory
    content = propp_fr.load_text_file("example")
    # Loads "example.txt" from current directory
    
    # Load a file with explicit extension
    content = propp_fr.load_text_file("data.txt")
    # Loads "data.txt" from current directory
    
    # Load from a specific directory
    content = propp_fr.load_text_file("story", files_directory="./texts")
    # Loads "./texts/story.txt"
    
    # Load a file with custom extension
    content = propp_fr.load_text_file("config", extension=".cfg")
    # Loads "config.cfg" from current directory
    
    # Full path example
    content = propp_fr.load_text_file("chapter1", files_directory="/data/books", extension=".md")
    # Loads "/data/books/chapter1.md"
    ```
    
    ### Notes
    
    - The function uses `pathlib.Path` for cross-platform path handling.
    - The function always reads files using UTF-8 encoding.
    - Extension is automatically added only if the filename doesn't already end with it.
    - The entire file content is loaded into memory at once, which may not be suitable for very large files.

