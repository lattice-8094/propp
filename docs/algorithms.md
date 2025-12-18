# Algorithms

## Mention Spans Detection Model

The mention detection module consists of two stacked BiLSTM-CRF models, each trained on a
different nesting level of mentions. During inference, predicted spans from both models are
combined. If two mention spans overlap, the span with the lower prediction confidence is
discarded.

### BERT Embeddings

The raw text is split into overlapping segments of length L (corresponding to the maximum
context window of the embedding model), with an overlap of L/2 to maximize the contextual
information available for each token. Each segment is processed by the CamemBERT-large
model, and the last hidden layer is extracted as token representations (1024 dimensions).

The final embedding for each token is computed as the average of its representations across
overlapping segments. The CamemBERT model is not fine-tuned for this task.

### BIOES Tag Prediction

For each sentence, token representations are passed through the BiLSTM-CRF model, which
outputs a sequence of BIOES tags:

- B-PER: Beginning of a mention
- I-PER: Inside a mention
- E-PER: End of a mention
- S-PER: Single-token mention
- O: Outside any mention

### Model Architecture

- Locked Dropout (0.5) applied to embeddings for regularization
- Projection Layer: Highway network mapping 1024 → 2048 dimensions
- BiLSTM Layer: Single bidirectional LSTM with 256 hidden units per direction
- Linear Layer: Maps 512-dimensional BiLSTM outputs to BIOES label scores
- CRF Layer: Enforces structured consistency in label predictions

### Model Training

- Data Splitting: Leave-One-Out Cross-Validation (LOOCV) with an 85% / 15% train-validation split
- Batch Size: 16 sentences per batch
- Optimization: Adam optimizer
  - Learning rate: 1.4 × 10⁻⁴
  - Weight decay: 1 × 10⁻⁵
- Learning Rate Scheduling: ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 2
- Average Training Epochs: 20
- Hardware: Trained on a single 6 GB NVIDIA RTX 1000 Ada Generation GPU

## Coreference Resolution Model

![Coreference Resolution Pipeline](images/coreference_resolution_pipeline_scheme.png)

Blog article about Coreference Resolution - Soon
Taking images from Cergy JE + MATE-SHS


![highest_ranked_mention](images/highest_ranked_mention.gif)

![coreference_resolution_scorring](images/coreference_resolution_scorring.gif)



### Research Articles
Antoine Bourgois and Thierry Poibeau. 2025.  
<a href="https://aclanthology.org/2025.crac-1.5/" target="_blank">The Elephant in the Coreference Room: Resolving Coreference in Full-Length French Fiction Works.</a>  
In *Proceedings of the Eighth Workshop on Computational Models of Reference, Anaphora and Coreference (CRAC 2025).* EMNLP 2025, Suzhou, China.  
<a href="https://arxiv.org/abs/2510.15594" target="_blank">arXiv</a>, <a href="https://hal.science/hal-05319970" target="_blank">HAL</a>, <a href="https://github.com/antoine-bourgois/antoine-bourgois.github.io/blob/main/articles/2025_CRAC_CoreferenceInFrenchNovels_poster.pdf" target="_blank">Poster</a>
