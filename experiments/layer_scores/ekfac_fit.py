"""
experiments.layer_scores.ekfac_fit
=====================================
Fit EK-FAC factors (A, G eigendecomposition + corrected eigenvalues Lambda)
for a ResNet-18/CIFAR-10 checkpoint, per the "Definitions" section of the
EK-FAC layer-scores spec in CLAUDE.md.

Device handling: --device {cuda,mps,cpu} controls where the forward/backward
passes run. All float64 work (torch.linalg.eigh via ekfac_math.eigh_factors,
which already runs in numpy/CPU) is unconditionally on CPU. Per-example
gradients for Lambda use forward/backward hooks (TorchScript vmap is not
used, per CLAUDE.md's Device handling section).
"""

import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from caffeine.kfac_collector import KFACCollector
from experiments.layer_scores.ekfac_math import eigh_factors
from experiments.layer_scores.ekfac_store import EKFACStore
from experiments.layer_scores.layer_spec import (
    LayerSpec,
    build_layer_specs,
    grad_W_from_patches,
    reshape_activations,
    reshape_grad_output,
)

log = logging.getLogger(__name__)


def resolve_device(requested: str) -> torch.device:
    """
    Resolve --device, falling back to cpu with a log line if the requested
    backend is unavailable. Never sets PYTORCH_ENABLE_MPS_FALLBACK.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        log.warning("mps requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def _sample_labels(logits: torch.Tensor, true_labels: torch.Tensor, fisher: str) -> torch.Tensor:
    """True Fisher (default): y_hat ~ p(y|x). --fisher empirical: true labels."""
    if fisher == "empirical":
        return true_labels
    probs = F.softmax(logits.detach(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def fit_A_G(
    model: nn.Module,
    specs: List[LayerSpec],
    loader,
    N_fac: int,
    device: torch.device,
    fisher: str = "true",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    E[a a^T], E[g g^T] via caffeine.kfac_collector.KFACCollector (unmodified) —
    model.eval() (frozen BN), no augmentation (caller passes an un-augmented
    loader), true-Fisher labels by default.
    """
    model.eval()
    layer_names = [s.name for s in specs]
    collector = KFACCollector(model, layer_names=layer_names, include_bias=True)
    criterion = nn.CrossEntropyLoss()

    seen = 0
    with collector:
        for x, y in loader:
            if seen >= N_fac:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            labels = _sample_labels(logits, y, fisher)
            loss = criterion(logits, labels)
            loss.backward()
            seen += x.shape[0]

    return {name: collector.factors(name) for name in layer_names}


class _PerExampleLambdaCollector:
    """
    Hook-based per-example gradient collector for the corrected eigenvalues:

        Lambda_l[i,j] = E_x[ (Q_G^T  grad_W(x)  Q_A)_{ij}^2 ]

    using the hook path (not torch.func.vmap), per CLAUDE.md's device section.
    Requires model.eval() (frozen BN) so per-example forward passes are truly
    independent and a single batched backward recovers exact per-example
    gradients after undoing the reduction='mean' scale (same trick as
    caffeine.kfac_collector.KFACCollector).
    """

    def __init__(self, model, specs: List[LayerSpec], eigs: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        self.specs = {s.name: s for s in specs}
        self.eigs = eigs  # name -> (Q_A, Q_G) torch tensors, already on the right device
        self._lambda_sum: Dict[str, torch.Tensor] = {}
        self._count: Dict[str, int] = {}
        self._fwd_cache: Dict[str, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        for name, spec in self.specs.items():
            self._handles.append(spec.module.register_forward_hook(self._fwd_hook(name)))
            self._handles.append(spec.module.register_full_backward_hook(self._bwd_hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def _fwd_hook(self, name):
        def hook(module, inputs, output):
            with torch.no_grad():
                self._fwd_cache[name] = reshape_activations(self.specs[name], inputs[0].detach())
        return hook

    def _bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            with torch.no_grad():
                go = grad_output[0].detach()
                N = go.shape[0]
                go = go * float(N)  # undo reduction='mean', per-example recovery
                g = reshape_grad_output(self.specs[name], go)
                a = self._fwd_cache.pop(name)
                M = a.shape[0]
                L = M // N
                Q_A, Q_G = self.eigs[name]
                lam_sum = self._lambda_sum.setdefault(
                    name, torch.zeros(Q_G.shape[0], Q_A.shape[0], dtype=torch.float64, device=a.device)
                )
                for n in range(N):
                    a_n = a[n * L:(n + 1) * L].double()
                    g_n = g[n * L:(n + 1) * L].double()
                    grad_W_n = grad_W_from_patches(self.specs[name], a_n, g_n, L)
                    C_n = Q_G.double().T @ grad_W_n @ Q_A.double()
                    lam_sum += C_n ** 2
                self._count[name] = self._count.get(name, 0) + N
        return hook

    def lambdas(self) -> Dict[str, np.ndarray]:
        return {
            name: (self._lambda_sum[name] / max(self._count[name], 1)).cpu().numpy().astype(np.float32)
            for name in self._lambda_sum
        }


def fit_lambda(
    model: nn.Module,
    specs: List[LayerSpec],
    loader,
    eigs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    N_corr: int,
    device: torch.device,
    fisher: str = "true",
) -> Dict[str, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    eigs_t = {
        name: (torch.from_numpy(Q_A).to(device), torch.from_numpy(Q_G).to(device))
        for name, (Q_A, Q_G) in eigs.items()
    }
    collector = _PerExampleLambdaCollector(model, specs, eigs_t)
    seen = 0
    with collector:
        for x, y in loader:
            if seen >= N_corr:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            labels = _sample_labels(logits, y, fisher)
            loss = criterion(logits, labels)
            loss.backward()
            seen += x.shape[0]
    return collector.lambdas()


def fit_ekfac(
    model: nn.Module,
    specs: List[LayerSpec],
    loader,
    N_fac: int,
    N_corr: int,
    device: torch.device,
    fisher: str = "true",
) -> Dict[str, dict]:
    """Full per-layer EK-FAC fit: A, G -> eigh (CPU/float64) -> Lambda."""
    AG = fit_A_G(model, specs, loader, N_fac, device, fisher=fisher)

    eigs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    result: Dict[str, dict] = {}
    for name, (A, G) in AG.items():
        Q_A, a, Q_G, gamma = eigh_factors(A, G)  # forced CPU/float64 internally
        eigs[name] = (Q_A, Q_G)
        result[name] = {"Q_A": Q_A, "a": a, "Q_G": Q_G, "gamma": gamma}

    Lambda = fit_lambda(model, specs, loader, eigs, N_corr, device, fisher=fisher)
    for name in result:
        result[name]["Lambda"] = Lambda[name]
        mean_lambda = float(np.mean(Lambda[name]))
        a, gamma = result[name]["a"], result[name]["gamma"]
        mean_outer = float(np.mean(np.outer(gamma, a)))
        log.info(
            "layer %-28s mean(Lambda)=%.4g  mean(gamma outer a)=%.4g",
            name, mean_lambda, mean_outer,
        )
    return result


def main():
    p = argparse.ArgumentParser(description="Fit EK-FAC factors for a trained ResNet-18/CIFAR-10 checkpoint")
    p.add_argument("--checkpoint", required=True, help="path to a train_resnet_cifar.py checkpoint .pt file")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--out", default="results/layer_scores/factors.h5")
    p.add_argument("--n-fac", type=int, default=50_000, help="examples for A/G fitting (TODO(neil): confirm default)")
    p.add_argument("--n-corr", type=int, default=10_000, help="TODO(neil): confirm default per-example-grad budget")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--fisher", choices=["true", "empirical"], default="true")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cpu")
    p.add_argument("--step", type=int, default=0, help="checkpoint step label for the store's key schema")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = resolve_device(args.device)

    from data.cifar10 import cifar10_loaders
    from models.resnet18 import resnet18_cifar

    model = resnet18_cifar(num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    train_loader, _, _ = cifar10_loaders(root=args.data_root, batch_size=args.batch_size, augment=False)
    specs = build_layer_specs(model)
    assert len(specs) == 21, f"expected 21 instrumented layers, got {len(specs)}"
    log.info("layers: %s", [s.name for s in specs])

    result = fit_ekfac(model, specs, train_loader, args.n_fac, args.n_corr, device, fisher=args.fisher)

    store = EKFACStore(args.out)
    total_bytes = 0
    for name, r in result.items():
        store.save_eigen(name, args.step, r["Q_A"], r["Q_G"], r["Lambda"], r["a"], r["gamma"], overwrite=True)
        layer_bytes = 4 * (r["Q_A"].size + r["Q_G"].size + r["Lambda"].size)
        total_bytes += layer_bytes
        log.info("saved %-28s bytes=%d", name, layer_bytes)
    log.info("total factor bytes: %d", total_bytes)


if __name__ == "__main__":
    main()
