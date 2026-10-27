"""
Tiny end-to-end integration dry run (synthetic in-memory data, not real
CIFAR-10 — avoids depending on a slow dataset download in CI/dev):
ekfac_fit -> queries -> scores -> e1/e2/e3, wired together, on a handful of
examples and queries. This is the "run on a tiny config before anything at
scale" check from Ground Rule #3 — it validates integration, not statistics.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.utils.data as tud

from experiments.layer_scores.ekfac_fit import fit_ekfac
from experiments.layer_scores.ekfac_math import full_bytes, layer_score, rotate_to_eigenbasis
from experiments.layer_scores.ekfac_store import EKFACStore
from experiments.layer_scores.layer_spec import build_layer_specs
from experiments.layer_scores.queries import Query
from experiments.layer_scores.scores import build_scores_table
from experiments.layer_scores.e1_concentration import compute_e1_tables, summary_table
from experiments.layer_scores.e2_variation import hot_sets_by_query, pairwise_jaccard


class _TinyCIFARLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 4, bias=True)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class _TensorDataset(tud.Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


@pytest.fixture
def tiny_setup(tmp_path):
    torch.manual_seed(0)
    n = 40
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, 4, (n,))
    ds = _TensorDataset(x, y)
    loader = tud.DataLoader(ds, batch_size=8, shuffle=False)

    model = _TinyCIFARLike().eval()
    device = torch.device("cpu")
    specs = build_layer_specs(model)
    assert len(specs) == 3  # conv1, conv2, fc

    result = fit_ekfac(model, specs, loader, N_fac=n, N_corr=n, device=device)

    store = EKFACStore(str(tmp_path / "factors.h5"))
    for name, r in result.items():
        store.save_eigen(name, 0, r["Q_A"], r["Q_G"], r["Lambda"], r["a"], r["gamma"], overwrite=True)

    eigs = {s.name: store.load_eigen(s.name, 0) for s in specs}
    return model, ds, specs, eigs, device


def test_integration_fit_scores_e1_e2(tiny_setup):
    model, ds, specs, eigs, device = tiny_setup

    queries = [
        Query(id="q0", type="R", size=3, seed=0, indices=[0, 1, 2]),
        Query(id="q1", type="R", size=3, seed=1, indices=[3, 4, 5]),
        Query(id="q2", type="C", size=4, seed=-1, indices=[0, 10, 20, 30]),
    ]

    df = build_scores_table(model, ds, queries, specs, eigs, alphas=[0.1], device=device)
    assert set(df["query_id"]) == {"q0", "q1", "q2"}
    assert (df["s_l"] >= 0).all()
    assert (df["bytes_l"] > 0).all()

    e1 = compute_e1_tables(df, alpha=0.1)
    assert (e1["k_q"] >= 1).all()
    assert (e1["k_q"] <= len(specs)).all()
    assert (e1["bytes_q"] <= 1.0 + 1e-9).all()
    summary = summary_table(e1)
    assert len(summary) > 0

    hot_sets = hot_sets_by_query(df, alpha=0.1, c=0.95)
    assert set(hot_sets.keys()) == {"q0", "q1", "q2"}
    query_type_of = df.drop_duplicates("query_id").set_index("query_id")["query_type"].to_dict()
    jac = pairwise_jaccard(hot_sets, query_type_of)
    assert len(jac) == 3  # 3 choose 2
    assert ((jac["jaccard"] >= 0) & (jac["jaccard"] <= 1)).all()
