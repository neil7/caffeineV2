# CLAUDE.md

This file gives Claude Code full context for this research project.
Read it completely before writing any code, suggesting changes, or answering questions.

---

## Project overview

This is a machine learning research project improving on **Koh & Liang (2017)**
("Understanding Black-box Predictions via Influence Functions", ICML 2017).

The core goal is **machine unlearning** — removing the influence of a specific
training data point from a trained model without retraining from scratch.

The original influence function formula is:

```
Δθ ≈ − H⁻¹ · ∇θ L(z, θ̂)
```

where H is the Hessian of the training loss, ∇θL(z, θ̂) is the gradient for
the point being forgotten, and Δθ is the resulting parameter update.

The bottleneck is H: for ResNet-18 (11M parameters), the full Hessian is an
11M × 11M matrix (~484 TB to store, O(d³) to invert). This project makes it
practical by storing compact Hessian approximations alongside training
checkpoints and paging them in on demand — directly analogous to how vLLM's
PagedAttention avoids pre-allocating the full KV cache.

---

## Architecture

### Two-tier storage (the central design decision)

```
HDF5 on disk  ←→  Redis in memory
(persistent)       (fast cache)
```

**HDF5** (`kfac_factors.h5`) stores Kronecker-factored (K-FAC) Hessian
approximations persistently, one pair of matrices per layer per checkpoint step.
Only the layers relevant to a deletion request are ever loaded — the same
"page in only what you need" property that makes vLLM efficient.

**Redis** caches recently computed results in memory with short TTLs so that
repeated or nearby deletion requests skip the expensive HDF5 read and matrix
inversion entirely.

### K-FAC approximation

For a linear layer `y = Wx`, the Hessian of the loss w.r.t. W factors as:

```
H_W  ≈  A ⊗ G
```

where:
- `A = E[x xᵀ]`  — input activation covariance  (d_in × d_in)
- `G = E[∇y ∇yᵀ]` — output gradient covariance  (d_out × d_out)

The key identity that makes this tractable without ever forming A⊗G:

```
(A ⊗ G)⁻¹ · vec(∇W)  =  vec( G⁻¹ · ∇W · A⁻ᵀ )
```

So the Newton step per layer reduces to two O(d³) matrix multiplications
instead of one O(d⁶) operation on the full Kronecker product.

### Block-diagonal approximation

The full Hessian also captures cross-layer interactions. The block-diagonal
approximation sets all off-diagonal blocks to zero — i.e. `∂²L/(∂Wₗ ∂Wₘ) = 0`
for `l ≠ m`. Each block then inverts independently, and a single-layer
unlearning request only needs to load that layer's A and G.

---

## Codebase

### Package layout

The original monolithic `main.py` (a.k.a. `kfac_unlearning.py`) is now the
installable package `kfac_unlearn/`, with PyTorch glue and experiments around it.

| Path | Contents |
|---|---|
| `kfac_unlearn/storage.py` | `HDF5KFACStore` + `_step_key`, `_chunk_shape` |
| `kfac_unlearn/serialization.py` | `_z_hash`, `_pack_array`, `_unpack_array` |
| `kfac_unlearn/cache.py` | `HVPCache` ABC + `InMemoryCache` (default) + `RedisHVPCache` + `make_cache` |
| `kfac_unlearn/kfac.py` | `kfac_newton_step` (pure numpy) |
| `kfac_unlearn/unlearner.py` | `KFACUnlearner` + `UnlearningResult` |
| `caffeine/layer_registry.py` | `instrumented_layers`, `weight_to_2d`, `weight_grad_2d` |
| `caffeine/kfac_collector.py` | `KFACCollector` — forward/backward hooks → A, G → HDF5 |
| `caffeine/grad_extract.py` | `extract_grad_W_per_layer`, `model_weights_per_layer`, `write_back_weights` |
| `models/`, `data/` | ResNet-18/CIFAR-10 and DistilBERT/Amazon wrappers + loaders + splits |
| `experiments/`, `benchmarks/` | train scripts, `unlearn` glue, metrics + retrain baseline |
| `scripts/run_demo.py` | self-contained demo — runs without torch or a live Redis server |

### Key classes

```python
HDF5KFACStore(filepath: str)
    .save_kfac(layer_name, step, A, G, overwrite=False)
    .load_kfac(layer_name, step) -> (A, G)
    .load_kfac_inverse(layer_name, step, damping=1e-4) -> (A_inv, G_inv)
    .nearest_checkpoint(layer_name, target_step) -> int
    .list_steps(layer_name) -> list[int]
    .list_layers() -> list[str]
    .storage_summary() -> dict

HVPCache(ABC)                          # the cache contract KFACUnlearner depends on
InMemoryCache(ttl_overrides)           # default backend — dependency-free, monotonic-clock TTL
RedisHVPCache(host, port, db, password, ttl_overrides)   # optional [redis] extra
    # both backends implement:
    .get_hvp(layer_name, z) -> ndarray | None
    .set_hvp(layer_name, z, delta_theta)
    .get_grad(z_id) -> ndarray | None
    .set_grad(z_id, grad)
    .get_hinv(layer_name, step) -> (A_inv, G_inv) | None
    .set_hinv(layer_name, step, A_inv, G_inv)
    .invalidate_layer(layer_name) -> int
    .cache_stats() -> dict
make_cache(cfg) -> HVPCache            # factory; backend selected by configs/storage.yaml

KFACUnlearner(store, cache, damping=1e-4)
    .unlearn(model_weights, grad_W_per_layer, z, z_id, checkpoint_step)
        -> UnlearningResult

kfac_newton_step(grad_W, A_inv, G_inv) -> delta_W   # standalone function
```

### HDF5 key schema

```
{layer_name}/{factor}/{step_key}

Examples:
  conv1/A/step_5000
  conv1/G/step_5000
  layer4.1/A/step_10000
  fc2/G/step_15000
  _meta/                   ← layer_dims and checkpoint_steps stored as JSON attrs
```

Datasets are stored as float32, chunked at `(min(d, 256), min(d, 256))`,
gzip-compressed at level 4.

### Redis key schema and TTLs

```
hvp:{layer_name}:{z_hash}    TTL 3600s  (1h)   — full Δθ for (layer, data point)
grad:{z_id}                  TTL 1800s (30m)   — raw gradient vector
hinv:{layer_name}:{step}     TTL 7200s  (2h)   — A⁻¹ and G⁻¹ (most expensive)
```

`z_hash` is the first 16 hex chars of SHA-256 of the serialised data point array.
All arrays are serialised with `_pack_array` (4-byte length prefix + JSON
metadata header + raw bytes) — 43 bytes overhead per array.

### Unlearning pipeline (cache lookup order)

```
1. Redis:  get hvp:{layer}:{z_hash}          → hit: apply Δθ immediately
2. Redis:  get hinv:{layer}:{step}           → hit: skip inversion
3. HDF5:   load_kfac(layer, nearest_step)    → load A, G from disk
           invert with damping λI
           cache result in Redis (hinv key)
4. Compute:  delta_W = −G⁻¹ @ grad_W @ A⁻ᵀ
             cache in Redis (hvp key)
5. Apply:    model_weights[layer] += delta_W
```

---

## Experimental setup

Two modalities are wired up (configs in `configs/`):

**1 · Vision — ResNet-18 / CIFAR-10** (`experiments/train_resnet_cifar.py`)
- ResNet-18 (~11M params, CIFAR stem: 3×3 stride-1 conv1, no maxpool, 10 classes)
- CIFAR-10 (3×32×32 RGB)
- Baseline accuracy: test 91.79%, val 92.02%, forget set 99.75%
- Unlearning time ~12.7s (pre-optimisation baseline); influence norm 0.0803
- 21 K-FAC layers: `conv1`, `layer{1..4}.{0,1}.conv{1,2}`, `layer{2..4}.0.downsample.0`, `fc`

**2 · NLP — DistilBERT / Amazon reviews** (`experiments/train_distilbert_amazon.py`)
- `DistilBertForSequenceClassification`, 5-star labels (`amazon_reviews_multi`)
- K-FAC over every `nn.Linear`: per-block `q/k/v/out_lin`, `lin1/lin2`,
  `pre_classifier`, `classifier`; embeddings + LayerNorm skipped
- Pad tokens masked out of A/G via `KFACCollector.set_token_mask`
- ⚠️ `amazon_reviews_multi` is delisted from the HF hub — `data/amazon.py` tries
  community mirrors and errors clearly if none resolve

Layer names follow PyTorch's `named_modules()` convention (dots, no slashes).

---

## Research context

### Position in the literature

This work sits at the intersection of two lines:

1. **Influence-function unlearning** (Koh & Liang 2017, Guo et al. 2020,
   Izzo et al. 2021, Qiao et al. ICLR 2025) — principled but computationally
   expensive due to Hessian operations.

2. **Hessian-free alternatives** (FUKD, MoDe, FUSED, FFMU) — cheaper but
   sacrifice the theoretical Newton-step guarantee.

The thesis of this project: the Hessian approach is the right approximation,
it just needs better infrastructure. K-FAC + paged storage makes it practical
without giving up the theoretical grounding.

### Key papers to keep in mind

- **Koh & Liang 2017** — the influence function formula this entire project
  is built on. The formula `Δθ ≈ −H⁻¹∇θL(z,θ̂)` is the north star.
- **Bourtoule et al. (Machine Unlearning, IEEE S&P 2021)** — exact unlearning
  via retraining / sharding; the expensive baseline we're beating.
- **Qiao et al. (Hessian-Free Online Certified Unlearning, ICLR 2025)** —
  most relevant recent work; explicitly positions itself as Hessian-free.
  Our work is positioned as "Hessian, but now actually feasible."
- **vLLM / PagedAttention (Kwon et al., SOSP 2023)** — the systems inspiration
  for the two-tier storage design. The analogy: KV cache : PagedAttention ::
  Hessian blocks : this project's HDF5+Redis paging scheme.
- **Ferrari (federated feature unlearning)** — a benchmark for comparison;
  also Hessian-based, restricted to image feature masking.

### Evaluation metrics (from "What Makes Unlearning Hard" paper)

1. **Forget accuracy** — how close is the unlearned model's behaviour on the
   forget set to a model retrained from scratch without that data.
2. **Retain accuracy / utility** — performance on the remaining data must not
   degrade.
3. **Unlearning time** — wall-clock time; must beat full retraining.
4. **Influence norm** — magnitude of Δθ; sanity-check that the update is bounded.

---

## Dependencies

Core (always required): `numpy`, `h5py`. Everything else is an optional extra so
the engine, demo, and tests run with no torch or Redis.

```
numpy, h5py            — core engine (float32 throughout)
[redis]                — RedisHVPCache backend (InMemoryCache is the default)
[torch]                — torch, torchvision (model loading, hooks, gradients)
[nlp]                  — transformers, datasets, scikit-learn (DistilBERT/Amazon)
[experiments]          — all of the above + pyyaml, tqdm
[dev]                  — pytest + torch + scikit-learn (the test suite)
```

Install:
```bash
pip install -e .                  # core
pip install -e ".[experiments]"   # full PyTorch modalities
pip install -e ".[dev]"           # to run pytest
```

Run the self-contained demo (no torch, no Redis needed):
```bash
python scripts/run_demo.py
```

---

## Conventions and constraints

### Data types
- All K-FAC factor matrices are stored and computed as **float32**.
- Use `np.asarray(x, dtype=np.float32)` when accepting external inputs.
- Never upcast to float64 inside the hot path — it doubles memory for no
  precision benefit at this scale.

### Damping
- Always add `damping * np.eye(d)` before inverting A or G.
- The code default is `1e-4`, but the Koh & Liang reference scripts use **`1e-2`
  for the CNN** (`influence-release/scripts/train_mnist.py`) and `0.0` for the
  convex logistic-regression case. Deep nets (ResNet, DistilBERT) need the larger
  value — `1e-4` is too small and produces oversized `delta_W`. Set damping
  **per experiment** in `configs/*.yaml` (both modalities default to `1e-2`);
  override at the call site, don't change the library default.
- If `delta_W` norms are unexpectedly large, increase damping first.
- Note `(A+λI) ⊗ (G+λI)` is not exactly `(A⊗G) + λI`; this per-factor damping is
  the standard K-FAC convention and differs slightly from the reference's
  `H + λI`.

### Layer naming
- Use PyTorch's `named_parameters()` convention: `conv1`, `layer1.0.conv1`,
  `layer4.1.bn2`, `fc`. The dot is a valid HDF5 group name character.
- Never use slashes in layer names passed to `HDF5KFACStore` — slashes are
  the HDF5 path separator and will be misinterpreted as group nesting.

### Serialisation
- Arrays going into Redis always go through `_pack_array` / `_unpack_array`.
  Do not use pickle — it is not safe across Python versions or numpy versions.
- The 43-byte overhead per array is negligible.

### Cache invalidation
- When a checkpoint is updated or replaced, call
  `cache.invalidate_layer(layer_name)` to evict stale inverse factors.
- The `hinv` TTL (2h) is intentionally longer than `hvp` (1h) because
  A⁻¹, G⁻¹ are reusable across many different data-point deletions at the
  same checkpoint step.

### Adding a new layer type (e.g. convolutional layers)
- Convolutional weights need to be reshaped to 2D before computing K-FAC
  factors: `W.reshape(C_out, -1)` where `-1` absorbs `C_in × kH × kW`.
- A becomes the covariance of the unrolled input patches (im2col output).
- G remains the covariance of output gradients, summed over spatial positions.
- The `kfac_newton_step` function itself is unchanged — it only sees 2D matrices.

### Testing
- Run `python scripts/run_demo.py` for the built-in round-trip demo
  (`numpy.random.default_rng(42)` for reproducibility; all asserts are
  load-bearing).
- Run `pytest tests/` for the unit suite: HDF5 round-trip, InMemory/Redis cache
  parity, `kfac_newton_step` vs dense `(A⊗G)⁻¹`, the full pipeline, and the
  **correctness gate** `test_kfac_vs_exact_hvp.py` (K-FAC influence vs exact
  full-Hessian influence — Pearson `r > 0.85` logreg, `r > 0.5` MLP).
- The gate asserts *correlation*, not equality: block-diagonal K-FAC is a
  deliberate approximation, not a bug.

---

## What to build next (open tasks)

- [x] **PyTorch hook integration** — `caffeine/kfac_collector.py` (`KFACCollector`)
      collects A/G via `register_forward_hook` + `register_full_backward_hook`
      and feeds `HDF5KFACStore.save_kfac`. Note the batch-N rescale that undoes
      `reduction='mean'`.
- [x] **ResNet-18 layer enumeration** — `caffeine/layer_registry.instrumented_layers`
      filters `named_modules()` to Linear/Conv2d (skips BatchNorm/ReLU/etc.).
- [x] **Gradient extraction for forget point** — `caffeine/grad_extract.py`
      (`extract_grad_W_per_layer` + `model_weights_per_layer` + `write_back_weights`).
- [~] **Benchmark harness** — `benchmarks/evaluate.py` (4 metrics + Pearson
      correlation) and `benchmarks/retrain_baseline.py` (retrain-from-scratch +
      `actual_loss_diffs`) exist; comparisons vs Ferrari/FUKD still TODO.
- [ ] **Checkpoint pruning policy** — decide which steps to keep in HDF5 as
      training progresses (keep last N, or every K steps, or by loss delta).
- [ ] **Federated setting** — each client runs `HDF5KFACStore` locally;
      server only sees `grad_W_per_layer`, never raw data. The K-FAC factors
      stay on-device. This is the privacy-preserving extension.

# Prompt: Per-layer EK-FAC score experiments for the Paged Curvature Store

## Context

You are working in Neil's repo for the Paged Curvature Store (SIGMOD target). The system stores per-layer EK-FAC curvature factors across a memory/disk hierarchy and serves them to a stream of deletion (forget) queries. Its fetch planner rests on one empirical premise: each forget query's Newton step is concentrated in a few layers, and which layers those are changes across queries. These experiments test that premise on ResNet-18 / CIFAR-10 before we build further.

Three experiments:

- **E1 Concentration.** How many layers, and how many bytes of factor state, does each query need to capture 90 / 95 / 99% of its update?
- **E2 Variation.** Do hot sets differ across queries? How does a static pinned layer set compare with a per-query set?
- **E3 Bound tightness.** Does a small always-resident sketch bound each layer's score tightly enough for a sound fetch planner?

E1 also answers a side question: how much of each query's update the output (`fc`) layer carries.

## Ground rules

1. **Read before writing.** Start by reading `CLAUDE.md` and `kfac_unlearning.py` (`HDF5KFACStore`, `RedisHVPCache`, `kfac_newton_step`, `KFACUnlearner`). Reuse what exists. Do not change existing behavior. Put new code in new modules under `experiments/layer_scores/`.
2. **Do not invent values.** Anything unknown (paths, checkpoint location, training recipe, SLURM partition, conda env) goes in the code as `TODO(neil): ...` and is listed in the final report. Do not guess.
3. **Correctness before scale.** Write the tests in the "Correctness tests" section first. Run them on a tiny config. If any test fails, stop and report. Do not run full experiments on unverified math.
4. **Traceability.** Every number in `REPORT.md` must come from a file in the results directory. Log the full config and seeds with every result file.
5. **No interpretation beyond the pre-registered thresholds** in the Report section. Do not make claims about LLMs.
6. **Model-agnostic core.** Write factor fitting and scoring against a layer registry (module list plus a reshape function), so a Pythia run can reuse it later. Do not implement LLM runs now.


## Step 0: Inspect and plan

Before writing code, report:

- what `kfac_unlearning.py` already provides, and whether it collects A and G via hooks (the overview lists hooks as an open task);
- where the trained ResNet-18 checkpoint lives (`TODO(neil)` if not found);
- how the CIFAR-10 data pipeline is set up;
- a short implementation plan mapped to the steps below.

Then proceed.

## Definitions (implement exactly)

**Layer set.** All `Conv2d` and `Linear` modules: `conv1`, 16 block convs, 3 downsample 1×1 convs, and `fc`. Expect 21. Assert the count and print the names in network order. BatchNorm parameters are excluded from the curvature store. Log their share of the forget-gradient norm separately.

**Shapes.** Reshape each weight to `W_l` of shape `(d_out, d_in)`. For conv layers, `d_in = C_in · k_h · k_w`. If a layer has a bias, append a homogeneous 1 to its input activations (`fc` has a bias; verify that the convs do not).

**Factors.** Compute these with `model.eval()` (frozen BN) and no augmentation. Use KFC for conv layers and standard K-FAC for linear layers.

- `A_l = E[ā āᵀ]`, where ā is the (unfolded, bias-augmented) input activation. The expectation runs over examples and spatial positions.
- `G_l = E[g gᵀ]`, where g is the gradient w.r.t. the layer output, per spatial position.
- Backward passes for G and Λ use labels sampled from the model, `ŷ ~ p(y|x)` (true Fisher). Add a flag `--fisher empirical` that uses true labels instead.

**Eigendecomposition.** Use `torch.linalg.eigh` in float64 and store the results in float32:

- `A_l = Q_A diag(a) Q_Aᵀ`
- `G_l = Q_G diag(γ) Q_Gᵀ`

`eigh` returns eigenvalues in ascending order. Reorder everything descending.

**Corrected eigenvalues.**

- `Λ_l[i,j] = E_x[ (Q_Gᵀ ∇_{W_l} L(x, ŷ) Q_A)_{ij}² ]`
- Use per-example gradients. For conv layers, `∇W = Σ_t g_t ā_tᵀ`, summed over spatial positions t.
- Λ has shape `(d_out, d_in)` and is ≥ 0. Its scale comes from the per-example gradients, so the A/G scaling convention does not affect scores.

**Damping.**

- Per-layer: `λ_l = α · mean(Λ_l)`, with α ∈ {0.01, 0.1, 1.0}. Default α = 0.1.
- Global variant: `λ = α · mean over all layers' Λ`.
- `TODO(neil)`: confirm the default α. Neil's recollection is that Grosse et al. use 0.1 × mean; this needs checking.

**Forget gradient.**

- `V_l = Σ_{z∈F} ∇_{W_l} L(z)`, using true labels, `model.eval()`, and no augmentation.
- Rotate it into the eigenbasis: `C_l = Q_Gᵀ V_l Q_A`.

**Layer score and full step.**

- Score: `s_l = Σ_ij C_ij² / (Λ_ij + λ_l)`.
- Full step for layer l: `Δθ*_l = Q_G [ C_l ⊘ (Λ_l + λ_l) ] Q_Aᵀ`.
- Full factor bytes: `bytes_l = 4 · (d_in² + d_out² + d_in · d_out)` (fp32 Q_A, Q_G, Λ).

**Rank-r sketch.**

- Ranks: `r_A = min(r, d_in)` and `r_G = min(r, d_out)`. For `fc`, `r_G ≤ 10`.
- `U_A` = top-r_A columns of Q_A. `U_G` = top-r_G columns of Q_G.
- `Λ_blk = Λ_l[top r_G rows, top r_A cols]`.
- `τ_l(r)` = the max of Λ_l over entries outside that block. For r = 0, `τ_l = max(Λ_l)`.
- Projection and masses: `C_r = U_Gᵀ V_l U_A`, `m_in = ‖C_r‖_F²`, `m_out = max(‖V_l‖_F² − m_in, 0)`.
- In-block score: `s_in = Σ C_r² / (Λ_blk + λ_l)`.
- Lower bound: `L_l = s_in + m_out / (τ_l + λ_l)`.
- Upper bound: `U_l = s_in + m_out / λ_l`.
- Sketch bytes: `4 · (r_A·d_in + r_G·d_out + r_A·r_G + 1)`.
- At full rank, `m_out ≈ 0` and `L_l = U_l = s_l`.

## Step 1: Model

Load the existing trained CIFAR ResNet-18 (`TODO(neil)`: checkpoint path). The architecture is a 3×3 stem (3→64), four stages of [2,2,2,2] BasicBlocks (64/128/256/512 channels), global average pooling, and `fc` 512→10.

Report test accuracy. The prior run reached about 91.8%; flag a large deviation. If no checkpoint exists, stop and ask. Do not pick a training recipe.

## Step 2: Fit EK-FAC

- Fit A and G on `N_fac` training examples. Default: all 50k, no augmentation.
- Fit Λ on `N_corr` per-example gradients. Default: 10k (`TODO(neil)`: confirm).
- Compute per-example gradients via forward/backward hooks (preferred) or `torch.func.vmap(grad(...))`.
- Store results in HDF5 via `HDF5KFACStore` if its schema fits. Otherwise use a new store with keys `/ckpt_{step}/{layer}/{Q_A, Q_G, Lambda, a, gamma}`.
- Log the bytes per layer and the total.
- Sanity print per layer: `mean(Λ_l)` against `mean(γ ⊗ a)`.

## Step 3: Forget queries

Use fixed seeds. Save query membership to `queries.json` with type, size and seed.

- **R (random):** uniform subsets of the train set, sizes {1, 10, 100, 1000}, 15 draws each (60 queries).
- **C (class):** each full class (10 queries), plus a random 10% of each class with 2 seeds (20 queries).
- **M (memorization proxy):** rank training points by per-example loss under the trained model, using true labels. Queries are the top-k for k ∈ {10, 100}, plus 18 random draws of size 100 from the top 1% (20 queries total). Label this a proxy in the report. True memorization scores need leave-out training (`TODO(neil)` if wanted).
- **N (neighborhood):** pick a seed point and take its k nearest neighbors in penultimate (512-d) feature space, for k ∈ {10, 100}, with 20 seeds each (40 queries). These sets are semantically coherent but not class-aligned.

That is about 150 queries in total.

## Step 4: Scores

For every query and α, compute per layer:

- `‖V_l‖_F` and `s_l`;
- the BN forget-gradient norm share;
- `s_l / Σ s`.

Write one row per (query, layer, α) to `scores.parquet`.

## E1: Concentration

For each query, for c ∈ {0.90, 0.95, 0.99}:

- `k_q(c)`: sort layers by `s_l` descending and count how many are needed to reach coverage c.
- `bytes_q(c)`: add layers greedily by `s_l / bytes_l` until coverage reaches c, then report the result as a fraction of total bytes.

Figures:

- **E1a:** CDF of `k_q(c)` per query type.
- **E1b:** CDF of `bytes_q(c)` per query type.
- **E1c:** heatmap of `s_l / Σ s`, with queries as rows grouped by type and layers as columns in network order.
- **E1d:** distribution of the `fc` share and the BN gradient-norm share, per query type.

Table: median and p90 of `k_q` and `bytes_q`, per type × c.

## E2: Variation

Let `H_q(c)` be the greedy byte-ordered hot set from E1.

**Overlap.** Compute pairwise Jaccard of hot sets within each query type and across types. Plot boxplots plus a type × type mean matrix.

**Static pinning baseline.**

1. Split the queries 50/50 by seed into a train half and a test half, stratified by type.
2. On the train half, build `S_static(c)`: add layers greedily by mean share / bytes until at least 95% of train queries reach coverage c.
3. On the test half, report two things:
   - the fraction of queries that reach coverage c using `S_static`;
   - `R(c) = bytes(S_static) / mean_q bytes_q(c)`.

`R ≈ 1` means per-query adaptivity buys nothing.

**Leave-one-type-out.** Build the static set on three query types and test it on the fourth. Report the same two numbers.

## E3: Bound tightness and a sound planner

Use r ∈ {0, 8, 16, 32, 64, 128, 256} and the default α. For each query, layer and r, record `U_l / L_l` and `m_out / ‖V_l‖_F²` in `bounds.parquet`.

**Sound planner.** For ε ∈ {0.01, 0.05, 0.10}:

1. Sort layers by `U_l / bytes_l` ascending.
2. Add layers to the skip set K while `Σ_{l∈K} U_l ≤ ε · Σ_all L_l`.
3. Fetch every layer not in K.

**Oracle.** Run the same greedy procedure using exact `s_l` with budget `ε · Σ s`.

Report:

- `bytes_fetched_sound / bytes_fetched_oracle`;
- total resident sketch bytes as a fraction of total full-factor bytes.

**Soundness assertion.** For every (query, r, ε), assert `Σ_{l∈K} s_l ≤ ε · Σ_all s_l`. Any violation is a bug. Stop and report.

Figures:

- **E3a:** median `U/L` vs r, per layer (log y).
- **E3b:** `m_out` fraction vs r, per layer.
- **E3c:** sound/oracle bytes ratio vs r, one line per ε.
- **E3d:** resident sketch size vs r.

## Correctness tests (pytest; write and pass these first)

- **T1, IHVP vs explicit.** For `fc` (513 × 10 = 5130 params), build the explicit damped EK-FAC matrix `(Q_G⊗Q_A) diag(vec Λ) (Q_G⊗Q_A)ᵀ + λI` and solve it directly in float64. Compare with the factored IHVP. Relative error must be < 1e-6. This test pins down the vec / Kronecker ordering convention.
- **T2, score identity.** `⟨vec V, IHVP(V)⟩` equals `Σ C² / (Λ + λ)`.
- **T3, additivity lemma.** For random layer subsets S, `‖Δθ_S − Δθ*‖²_H` equals `Σ_{l∉S} s_l` to 1e-5 relative. Here H is the block-diagonal damped EK-FAC, applied via factored HVPs.
- **T4, bounds.** `L ≤ s ≤ U` for every layer, for random V and several r. At full rank, `L = U = s`.
- **T5, per-example gradients.** The sum of per-example gradients equals the summed batch gradient for every layer.
- **T6, eigenbases.** Eigenvalues are in descending order, and `‖QᵀQ − I‖ < 1e-5`.
- **T7, layer count.** There are 21 layers, and their names are printed.

## Compute (Talapas, SLURM)

- Write sbatch templates in `experiments/layer_scores/slurm/`.
- `TODO(neil)`: partition, account, GPU type, conda env, data path.
- Everything should fit on one GPU. Print a runtime estimate after the tiny-config run.
- Cache factors and per-query results so jobs resume after preemption.

## Device handling

- CLI flag `--device {cuda,mps,cpu}` on every entrypoint that touches a model.
- All float64 work (eigendecompositions, correctness tests T1–T3) runs on CPU
  regardless of `--device` — MPS/CUDA float64 support is unreliable/absent.
- Do not set `PYTORCH_ENABLE_MPS_FALLBACK`. If an op is unsupported on MPS,
  move that step to CPU explicitly and log it (layer name + op) — no silent
  automatic fallback.
- Per-example gradients (Λ fitting, T5) use the hook-based path, not
  `torch.func.vmap(grad(...))`.

## Outputs

Code layout:

```
experiments/layer_scores/
  ekfac_fit.py  queries.py  scores.py
  e1_concentration.py  e2_variation.py  e3_bounds.py  plots.py
  tests/  slurm/
```

Results layout:

```
results/layer_scores/<run_id>/
  config.yaml  factors.h5  queries.json
  scores.parquet  bounds.parquet  figures/*.pdf  REPORT.md
```

## REPORT.md

`REPORT.md` contains:

- what ran, runtimes, and any deviations from this spec;
- all tables and figure references;
- the open `TODO(neil)` list;
- the quadrant classification below.

**Quadrant classification.** Use these pre-registered thresholds:

- **Concentrated** if the median over queries of `bytes_q(0.95)` is ≤ T_conc of total bytes. Proposed T_conc = 0.30 (`TODO(neil)`: confirm).
- **Variable** if `R(0.95)` is ≥ T_var on the held-out half. Proposed T_var = 1.5 (`TODO(neil)`: confirm).

State the resulting quadrant: (concentrated, variable), (concentrated, static), (diffuse, variable), or (diffuse, static). Add no other interpretation.