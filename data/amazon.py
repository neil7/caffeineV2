"""
data.amazon
==========
Amazon-reviews loader for the DistilBERT modality.

Dataset: `amazon_reviews_multi` (5-star, multilingual). ⚠️ This was removed from
the HuggingFace hub at Amazon's request, so we try a sequence of community
mirrors and raise a clear error if none resolve. Field `review_body` is the
text; `stars` ∈ {1..5} → label = stars - 1 (0..4), matching num_labels=5.

`datasets`/`transformers` are imported lazily so `import data.amazon` works
without the `[nlp]` extra installed.
"""

from typing import Optional

# Mirrors tried in order — the canonical name is delisted.
_MIRRORS = [
    ("mteb/amazon_reviews_multi", "en"),
    ("defunct-datasets/amazon_reviews_multi", "en"),
    ("amazon_reviews_multi", "en"),
]


def _load_raw(subset_train: Optional[int], subset_test: Optional[int]):
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Amazon modality needs the `datasets` library. "
            "Install with: pip install kfac-unlearn[nlp]"
        ) from e

    last_err = None
    for name, config in _MIRRORS:
        try:
            ds = load_dataset(name, config)
            break
        except Exception as e:  # mirror missing / renamed
            last_err = e
            ds = None
    if ds is None:
        raise RuntimeError(
            "Could not load `amazon_reviews_multi` from any known mirror "
            f"({[m[0] for m in _MIRRORS]}). It was delisted from the HF hub; "
            "point this loader at a local copy or another mirror. "
            f"Last error: {last_err}"
        )

    train = ds["train"]
    test = ds["test"] if "test" in ds else ds["validation"]
    if subset_train:
        train = train.select(range(min(subset_train, len(train))))
    if subset_test:
        test = test.select(range(min(subset_test, len(test))))
    return train, test


def amazon_loaders(
    tokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    subset_train: Optional[int] = 50_000,
    subset_test: Optional[int] = 10_000,
):
    """
    Return (train_loader, test_loader, train_dataset). `tokenizer` is a
    DistilBERT tokenizer (see models.distilbert.distilbert_amazon).

    subset_* default to tractable slices; pass None for the full dataset.
    """
    import torch
    from transformers import DataCollatorWithPadding

    train, test = _load_raw(subset_train, subset_test)

    def tokenize(batch):
        enc = tokenizer(batch["review_body"], truncation=True, max_length=max_length)
        enc["labels"] = [s - 1 for s in batch["stars"]]   # stars 1..5 -> 0..4
        return enc

    cols_to_remove = train.column_names
    train = train.map(tokenize, batched=True, remove_columns=cols_to_remove)
    test = test.map(tokenize, batched=True, remove_columns=cols_to_remove)
    train.set_format("torch")
    test.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    return train_loader, test_loader, train
