import os
import json
from collections import Counter

import torch
import torch.nn as nn

from importlib import resources

ATTRIBUTE_COLUMNS = ["char_att_agent", "char_att_patient", "char_att_mod", "char_att_poss"]


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class Classifier(nn.Module):
    def __init__(self, input_dim=1024, dropout_rate=0.5, hidden_layer_width=512, num_classes=17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_width),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_width, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ontology_classification_model(model_dir: str = None, device: torch.device = None):
    if model_dir is None:
        model_dir = str(resources.files("propp_fr.data").joinpath("annotated_attributes_data"))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    Load the trained ontology classifier and its label mapping.

    Parameters
    ----------
    model_dir : str
        Directory containing ``attributes_classification_model.pt`` and
        ``training_label_mapping.json``.
    device : torch.device, optional
        Defaults to CUDA if available, otherwise CPU.

    Returns
    -------
    model : Classifier
        Model in eval mode, moved to ``device``.
    id2label : dict
        Mapping from predicted integer index to string label.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- model weights --
    checkpoint_path = os.path.join(model_dir, "attributes_classification_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Classifier(
        input_dim=checkpoint["input_dim"],
        hidden_layer_width=checkpoint["hidden_layer_width"],
        dropout_rate=checkpoint["dropout_rate"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # -- label mapping --
    mapping_path = os.path.join(model_dir, "training_label_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    # ---- load mappings ----
    with open(os.path.join(model_dir, "syntactic_role_mapping.json"), "r", encoding="utf-8") as f:
        syntactic_role_mapping = json.load(f)

    return {'model': model,
            'id2label': id2label,
            'syntactic_role_mapping': syntactic_role_mapping}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_attributes(tokens_df, tokens_embedding_tensor: torch.Tensor, model_dict):
    """
    Classify character attributes for all attribute-bearing tokens and write
    predictions back into ``tokens_df``.

    Parameters
    ----------
    tokens_df : pd.DataFrame
        Token-level dataframe. Must contain the four ``char_att_*`` columns.
    tokens_embedding_tensor : torch.Tensor
        Shape ``(num_tokens, embedding_dim)``. Row i corresponds to row i of
        ``tokens_df``.
    model : Classifier
        Trained model returned by ``load_ontology_classification_model``.
    id2label : dict
        Label mapping returned by ``load_ontology_classification_model``.

    Returns
    -------
    pd.DataFrame
        ``tokens_df`` with a new ``ontology_{num_classes}`` column populated
        for attribute rows and ``None`` elsewhere.
    """
    model = model_dict["model"]
    id2label = model_dict["id2label"]
    syntactic_role_mapping = model_dict["syntactic_role_mapping"]

    device = next(model.parameters()).device
    num_classes = len(id2label)
    output_column = f"ontology_{num_classes}"

    # -- identify attribute-bearing rows --
    attribute_mask = (tokens_df[ATTRIBUTE_COLUMNS] != -1).any(axis=1)
    attribute_ids = tokens_df[attribute_mask].index.tolist()

    tokens_df[output_column] = None

    if not attribute_ids:
        return tokens_df

    # -- build embedding slice --
    attribute_embeddings = tokens_embedding_tensor[attribute_ids]  # (N, emb_dim)

    # -- append syntactic role vector if the model was trained with it --
    expected_input_dim = next(model.net.parameters()).shape[1]
    if expected_input_dim != attribute_embeddings.shape[1]:
        role_vectors = tokens_df.loc[attribute_ids, ATTRIBUTE_COLUMNS].copy()
        role_vectors = role_vectors.where(role_vectors == -1, 1).replace(-1, 0)
        role_tensor = torch.tensor(role_vectors.values.tolist(), dtype=attribute_embeddings.dtype)
        attribute_embeddings = torch.cat((attribute_embeddings, role_tensor), dim=1)

    # -- inference --
    with torch.no_grad():
        logits = model(attribute_embeddings.to(device))
        predicted_ids = logits.argmax(dim=1).cpu().numpy()

    predicted_labels = [id2label[i] for i in predicted_ids]

    tokens_df.loc[attribute_ids, output_column] = predicted_labels

    return tokens_df