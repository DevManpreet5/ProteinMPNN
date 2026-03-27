import torch
import triton
import triton.language as tl


@triton.jit
def virtual_cb_kernel(X_ptr, Cb_ptr, B, L, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    bid = pid // L
    lid = pid % L

    if bid >= B or lid >= L:
        return

    base = bid * L * 4 * 3 + lid * 4 * 3

    N0 = tl.load(X_ptr + base + 0 * 3 + 0)
    N1 = tl.load(X_ptr + base + 0 * 3 + 1)
    N2 = tl.load(X_ptr + base + 0 * 3 + 2)
    CA0 = tl.load(X_ptr + base + 1 * 3 + 0)
    CA1 = tl.load(X_ptr + base + 1 * 3 + 1)
    CA2 = tl.load(X_ptr + base + 1 * 3 + 2)
    C0 = tl.load(X_ptr + base + 2 * 3 + 0)
    C1 = tl.load(X_ptr + base + 2 * 3 + 1)
    C2 = tl.load(X_ptr + base + 2 * 3 + 2)

    b0 = CA0 - N0
    b1 = CA1 - N1
    b2 = CA2 - N2
    c0 = C0 - CA0
    c1 = C1 - CA1
    c2 = C2 - CA2

    a0 = b1 * c2 - b2 * c1
    a1 = b2 * c0 - b0 * c2
    a2 = b0 * c1 - b1 * c0

    Cb0 = -0.58273431 * a0 + 0.56802827 * b0 - 0.54067466 * c0 + CA0
    Cb1 = -0.58273431 * a1 + 0.56802827 * b1 - 0.54067466 * c1 + CA1
    Cb2 = -0.58273431 * a2 + 0.56802827 * b2 - 0.54067466 * c2 + CA2

    out_base = bid * L * 3 + lid * 3
    tl.store(Cb_ptr + out_base + 0, Cb0)
    tl.store(Cb_ptr + out_base + 1, Cb1)
    tl.store(Cb_ptr + out_base + 2, Cb2)


@triton.jit
def rbf_kernel(D_ptr, RBF_ptr, D_min, D_max, D_count, B, L, K, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    bid = pid // (L * K)
    rem = pid % (L * K)
    lid = rem // K
    kid = rem % K

    if bid >= B or lid >= L or kid >= K:
        return

    D_sigma = (D_max - D_min) / D_count
    D_val = tl.load(D_ptr + bid * L * K + lid * K + kid)

    for rbf_idx in range(D_count):
        D_mu = D_min + (D_max - D_min) * rbf_idx / (D_count - 1)
        diff = (D_val - D_mu) / D_sigma
        rbf_val = tl.exp(-diff * diff)
        tl.store(RBF_ptr + bid * L * K * D_count + lid * K * D_count + kid * D_count + rbf_idx, rbf_val)


@triton.jit
def fused_rbf_dist_kernel(A_ptr, B_ptr, E_idx_ptr, RBF_ptr, D_min, D_max, D_count, B, L, K, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    bid = pid // (L * K)
    rem = pid % (L * K)
    lid = rem // K
    kid = rem % K

    if bid >= B or lid >= L or kid >= K:
        return

    neighbor_idx = tl.load(E_idx_ptr + bid * L * K + lid * K + kid)
    neighbor_idx = neighbor_idx.to(tl.int32)

    dist_sq = 0.0
    for d in range(3):
        a_val = tl.load(A_ptr + bid * L * 3 + lid * 3 + d)
        b_val = tl.load(B_ptr + bid * L * 3 + neighbor_idx * 3 + d)
        diff = a_val - b_val
        dist_sq += diff * diff

    dist = tl.sqrt(dist_sq + 1e-6)
    D_sigma = (D_max - D_min) / D_count
    base_idx = bid * L * K * D_count + lid * K * D_count + kid * D_count

    for rbf_idx in range(D_count):
        D_mu = D_min + (D_max - D_min) * rbf_idx / (D_count - 1)
        diff = (dist - D_mu) / D_sigma
        rbf_val = tl.exp(-diff * diff)
        tl.store(RBF_ptr + base_idx + rbf_idx, rbf_val)


def virtual_cb(X):
    B, L, _, _ = X.shape
    Cb = torch.empty((B, L, 3), device=X.device, dtype=X.dtype)
    grid = lambda meta: (B * L,)
    virtual_cb_kernel[grid](X, Cb, B, L, BLOCK=256)
    return Cb


def rbf(D, D_min=2.0, D_max=22.0, D_count=16):
    B, L, K = D.shape
    RBF = torch.empty((B, L, K, D_count), device=D.device, dtype=D.dtype)
    grid = lambda meta: (B * L * K,)
    rbf_kernel[grid](D, RBF, D_min, D_max, D_count, B, L, K, BLOCK=256)
    return RBF


def fused_rbf_dist(A, B, E_idx, D_min=2.0, D_max=22.0, D_count=16):
    B_size, L, _ = A.shape
    K = E_idx.shape[-1]
    RBF = torch.empty((B_size, L, K, D_count), device=A.device, dtype=A.dtype)
    grid = lambda meta: (B_size * L * K,)
    fused_rbf_dist_kernel[grid](A, B, E_idx, RBF, D_min, D_max, D_count, B_size, L, K, BLOCK=256)
    return RBF
