"""kfac_newton_step matches the dense (A⊗G)⁻¹ Newton step on small matrices."""

import numpy as np

from kfac_unlearn import kfac_newton_step


def test_kfac_equals_dense_kron_inverse(rng):
    d_in, d_out = 6, 4
    rA = rng.standard_normal((d_in, d_in))
    rG = rng.standard_normal((d_out, d_out))
    A = rA.T @ rA + 0.1 * np.eye(d_in)          # SPD
    G = rG.T @ rG + 0.1 * np.eye(d_out)
    grad_W = rng.standard_normal((d_out, d_in))

    A_inv = np.linalg.inv(A)
    G_inv = np.linalg.inv(G)

    # K-FAC factored Newton step
    delta_kfac = kfac_newton_step(grad_W, A_inv, G_inv)

    # Dense: -(A⊗G)⁻¹ vec_F(grad_W), reshaped column-major
    H = np.kron(A, G)                            # (d_in*d_out, d_in*d_out)
    vecF = grad_W.flatten(order="F")
    delta_dense = -(np.linalg.inv(H) @ vecF).reshape((d_out, d_in), order="F")

    assert np.allclose(delta_kfac, delta_dense, atol=1e-8)


def test_shape_and_sign(rng):
    grad_W = np.ones((3, 5))
    A_inv = np.eye(5)
    G_inv = np.eye(3)
    delta = kfac_newton_step(grad_W, A_inv, G_inv)
    assert delta.shape == (3, 5)
    # with identity inverses, delta = -grad_W
    assert np.allclose(delta, -grad_W)
