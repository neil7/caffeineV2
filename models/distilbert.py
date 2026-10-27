"""
models.distilbert
================
DistilBERT sequence-classification wrapper for the Amazon-reviews modality.

K-FAC-eligible layers are the nn.Linear modules inside each transformer block
(attention q/k/v/out projections + FFN lin1/lin2) plus the classification head
(pre_classifier, classifier). Embeddings and LayerNorm are skipped — they have
no W = A ⊗ G structure.

transformers is imported lazily so `import models.distilbert` works even when
the `[nlp]` extra is not installed.
"""

from typing import List, Tuple

import torch.nn as nn

from caffeine.layer_registry import instrumented_layers


def distilbert_amazon(
    num_labels: int = 5,
    model_name: str = "distilbert-base-uncased",
) -> Tuple[nn.Module, object]:
    """
    Return (model, tokenizer). num_labels=5 matches amazon_reviews_multi stars.
    Requires `pip install kfac-unlearn[nlp]`.
    """
    try:
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "DistilBERT modality needs transformers. "
            "Install with: pip install kfac-unlearn[nlp]"
        ) from e

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    return model, tokenizer


def kfac_layers(model: nn.Module) -> List[str]:
    """All nn.Linear layers (attention + FFN + classifier head)."""
    return [name for name, _ in instrumented_layers(model)]
