# K-FAC Machine Unlearning — *Hessian, but now feasible*

Efficient **machine unlearning** via influence functions, made practical with a
Kronecker-factored (K-FAC) Hessian approximation and two-tier *paged* storage.

This project improves on **Koh & Liang (2017)**, *"Understanding Black-box
Predictions via Influence Functions"* (ICML 2017). The goal: remove the
influence of a specific training point from a trained model **without retraining
from scratch**, using the influence-function update

```
Δθ ≈ − H⁻¹ ∇θ L(z, θ̂)
```

The bottleneck is `H`. For ResNet-18 (~11M params) the full Hessian is 11M×11M
(~484 TB, `O(d³)` to invert). We make it tractable by storing compact K-FAC
factors alongside checkpoints and **paging in only the layers a deletion
touches** — the same insight behind vLLM's PagedAttention.

---

## Architecture

```
        HDF5 on disk            ←→            cache in memory
   (persistent K-FAC factors)        (InMemoryCache | RedisHVPCache)
       page in per-layer                 skip re-inversion / re-HVP
```

**K-FAC.** For a layer `y = Wx`, the Hessian block factors as `H_W ≈ A ⊗ G` with
`A = E[xxᵀ]` (input covariance) and `G = E[∇y∇yᵀ]` (output-grad covariance). The
identity

```
(A ⊗ G)⁻¹ · vec(∇W) = vec( G⁻¹ · ∇W · A⁻ᵀ )
```

turns the Newton step into two `O(d³)` matmuls — never forming `A⊗G`.

**Block-diagonal.** Off-diagonal (cross-layer) blocks are set to zero, so each
layer inverts independently and a single-layer deletion reads only that layer's
`A`, `G` from HDF5.

**Package map**

| Module | Contents |
|---|---|
| `kfac_unlearn/storage.py`   | `HDF5KFACStore` — persistent chunked factor store |
| `kfac_unlearn/cache.py`     | `HVPCache` ABC + `InMemoryCache` (default) + `RedisHVPCache` |
| `kfac_unlearn/kfac.py`      | `kfac_newton_step` — core math (pure numpy) |
| `kfac_unlearn/unlearner.py` | `KFACUnlearner` pipeline + `UnlearningResult` |
| `caffeine/`                 | PyTorch glue: hooks (`KFACCollector`), grad extraction |
| `models/`, `data/`          | ResNet-18/CIFAR-10 and DistilBERT/Amazon wrappers + loaders |
| `experiments/`, `benchmarks/` | train, unlearn, and evaluate end-to-end |

---

## Installation

The core engine runs on **numpy + h5py only** — a reviewer can install and run
the demo and tests with no torch or Redis.

```bash
pip install -e .                  # core: numpy + h5py
pip install -e ".[experiments]"   # torch, torchvision, transformers, datasets, ...
pip install -e ".[redis]"         # optional Redis cache backend
pip install -e ".[dev]"           # pytest + torch + scikit-learn (runs the test suite)
```

No Redis server is required — `InMemoryCache` is the default backend.

---

## Quickstart (synthetic, no GPU, no Redis)

```bash
python scripts/run_demo.py
```

Round-trips K-FAC factors through HDF5, computes a Newton step, and checks the
cache serialisation — all on synthetic numpy data.

---

## Experiments

Two modalities, configured by YAML in `configs/`.

### (a) ResNet-18 on CIFAR-10

```bash
python scripts/train.py   --config configs/resnet18_cifar10.yaml
python scripts/unlearn.py --config configs/resnet18_cifar10.yaml
```

### (b) DistilBERT on Amazon reviews

```bash
python scripts/train.py   --config configs/distilbert_amazon.yaml
python scripts/unlearn.py --config configs/distilbert_amazon.yaml
```

> **Dataset note.** `amazon_reviews_multi` was removed from the HuggingFace hub
> at Amazon's request. `data/amazon.py` tries community mirrors and raises a
> clear error if none resolve — point it at a local copy if needed.

Forget sets support both a **random subset** (default) and **whole-class**
forgetting (`split.forget_mode` in the config).

---

## Benchmarks

```bash
python scripts/benchmark.py --config configs/resnet18_cifar10.yaml
```

Reports the four metrics from *"What Makes Unlearning Hard"*: **forget accuracy**
(vs a retrain-from-scratch baseline), **retain accuracy**, **unlearning time**,
and **influence norm** — plus a Pearson-correlation check of predicted vs actual
loss change that mirrors Koh & Liang's `test_retraining`.

---

## Correctness

We validate against the original influence definition (`influence-release/`, the
vendored Koh & Liang TensorFlow reference). `tests/test_kfac_vs_exact_hvp.py`
compares K-FAC influence to the **exact full-Hessian** influence:

- single Linear (logistic regression): Pearson `r > 0.85`
- 2-layer MLP (block-diagonal approximation active): `r > 0.5`

Block-diagonal K-FAC is a deliberate approximation, so the gate asserts strong
*correlation*, not numeric equality. Run the suite:

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Citing

```bibtex
@misc{kfac-unlearn,
  title  = {K-FAC Machine Unlearning: Hessian, but now feasible},
  author = {Sharma, Neil},
  year   = {2026}
}
```

Builds directly on:

- Koh & Liang, *Understanding Black-box Predictions via Influence Functions*, ICML 2017.
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM), SOSP 2023.
- Bourtoule et al., *Machine Unlearning*, IEEE S&P 2021.
- Qiao et al., *Hessian-Free Online Certified Unlearning*, ICLR 2025.

---

## License & vendored code

`influence-release/` is the unmodified Koh & Liang reference (its own
`LICENSE.md`). The rest of this repository is MIT-licensed.
