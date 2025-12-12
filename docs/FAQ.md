# Frequent Errors

```python
spacy_model, mentions_detection_model, coreference_resolution_model = load_models()

----> ValueError: Cannot use GPU, CuPy is not installed
```

For spacy to run on the GPU you need to install CuPy compatible with your installed CUDA version.

To know CUDA version you can get

```bash
nvidia-smi
```



CUDA Version: 12.5
