"""
Correctness tests T5 (per-example gradients sum to the batch gradient) and T7
(layer count == 21) from the EK-FAC layer-scores spec. Requires torch.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from experiments.layer_scores.layer_spec import (
    build_layer_specs,
    grad_W_from_patches,
    reshape_activations,
    reshape_grad_output,
)

torch.manual_seed(17)


class _TinyNet(nn.Module):
    """Tiny Conv2d + Linear stack mirroring resnet18's instrumented-layer shapes."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 3, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_T7_layer_count():
    from models.resnet18 import resnet18_cifar

    model = resnet18_cifar(num_classes=10)
    specs = build_layer_specs(model)
    names = [s.name for s in specs]
    print("layer names (network order):", names)
    assert len(specs) == 21, f"expected 21 instrumented layers, got {len(specs)}: {names}"


def test_T5_per_example_grad_sums_to_batch_grad():
    model = _TinyNet().eval()  # eval: no BN/dropout here, but matches spec's frozen-stats convention
    specs = build_layer_specs(model)
    spec_by_name = {s.name: s for s in specs}

    N = 6
    x = torch.randn(N, 3, 5, 5)
    y = torch.randint(0, 3, (N,))

    fwd_cache = {}
    handles = []
    for s in specs:
        def fwd_hook(module, inputs, output, _name=s.name):
            fwd_cache[_name] = reshape_activations(spec_by_name[_name], inputs[0].detach())
        handles.append(s.module.register_forward_hook(fwd_hook))

    bwd_cache = {}
    for s in specs:
        def bwd_hook(module, grad_input, grad_output, _name=s.name):
            go = grad_output[0].detach() * float(N)  # undo reduction='mean'
            bwd_cache[_name] = reshape_grad_output(spec_by_name[_name], go)
        handles.append(s.module.register_full_backward_hook(bwd_hook))

    model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model(x), y)
    loss.backward()

    for h in handles:
        h.remove()

    for s in specs:
        a = fwd_cache[s.name]
        g = bwd_cache[s.name]
        M = a.shape[0]
        L = M // N
        per_example_sum = torch.zeros(s.d_out, s.d_in)
        for n in range(N):
            a_n = a[n * L:(n + 1) * L]
            g_n = g[n * L:(n + 1) * L]
            per_example_sum += grad_W_from_patches(s, a_n, g_n, L)

        # batch gradient (undo reduction='mean'): module.weight.grad * N
        batch_grad = s.module.weight.grad.detach() * float(N)
        batch_grad_2d = batch_grad.reshape(s.d_out, -1)
        if s.has_bias:
            bias_grad = (s.module.bias.grad.detach() * float(N)).reshape(s.d_out, 1)
            batch_grad_2d = torch.cat([batch_grad_2d, bias_grad], dim=1)

        rel_err = (per_example_sum - batch_grad_2d).norm() / batch_grad_2d.norm().clamp_min(1e-12)
        assert rel_err < 1e-4, f"T5 failed for layer {s.name}: rel_err={rel_err.item()}"
