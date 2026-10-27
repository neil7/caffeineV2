"""
scripts/benchmark.py
===================
Run the unlearning benchmark suite: the four CLAUDE.md metrics plus the
predicted-vs-actual influence correlation against a retrain baseline.

    python scripts/benchmark.py --config configs/resnet18_cifar10.yaml

This orchestrates benchmarks.evaluate + benchmarks.retrain_baseline; it expects
a trained checkpoint and is intended to be filled in per modality (the forward
function and data loaders differ between vision and NLP).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser(description="Unlearning benchmark suite")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = _load_yaml(args.config)
    print(f"Benchmark config: {cfg['modality']}")
    print(
        "Wire up loaders + a `forward(model, batch, device)` for this modality, "
        "then call:\n"
        "  benchmarks.evaluate.evaluate(...)            # forget/retain/test acc\n"
        "  benchmarks.evaluate.influence_norm(results)  # ‖Δθ‖\n"
        "  benchmarks.retrain_baseline.retrain_on_retain(...)  # gold standard\n"
        "  benchmarks.evaluate.correlation_check(predicted, actual)  # faithfulness"
    )


if __name__ == "__main__":
    main()
