import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# =============================================================================
# Idea
# -----------------------------------------------------------------------------
# The termination-by-combination step maps the outer product
#   PP_mat = P_vec @ P_vec.T            (shape [N, N])
# onto a dead-chain distribution
#   D[ii + jj + 1] += k_tc * PP_mat[ii, jj]     (lower triangle, ii + jj + 2 <= N)
#
# The *structure* of that mapping (which PP_mat entry lands in which D bin)
# depends only on N and k_tc -- NOT on P. So we build a constant operator T
# ONCE, then every step is a single sparse matrix-vector product:
#
#   D = T @ vec(PP_mat)
#
# T has shape [N x N^2] (it maps the length-N^2 flattened PP_mat to length-N D),
# and it is extremely sparse: one nonzero per valid (ii, jj) pair. Reusing it in
# a loop where P keeps changing avoids rebuilding masks / bincount every step.
# =============================================================================


def build_termination_operator(N, k_tc=1.0):
    """Constant [N x N^2] sparse operator T with D = T @ PP_mat.reshape(-1).

    Uses C-order flattening, so PP_mat[ii, jj] sits at column ii * N + jj.
    """
    ii, jj = np.indices((N, N))
    s = ii + jj

    # Same selection as the loop: lower triangle and within range.
    valid = (ii >= jj) & (s + 2 <= N)

    rows = (s[valid] + 1)                      # target D bin
    cols = ii[valid] * N + jj[valid]           # column in flattened PP_mat (C order)
    data = np.full(rows.shape, k_tc, dtype=float)

    T = sp.csr_matrix((data, (rows, cols)), shape=(N, N * N))
    return T


def build_overflow_mask(N):
    """Boolean [N^2] mask (C order) for the entries that overflow into too_high."""
    ii, jj = np.indices((N, N))
    overflow = (ii >= jj) & (ii + jj + 2 > N)
    return overflow.reshape(-1)


if __name__ == "__main__":
    N = 20
    k_tc = 1.0

    # ---- Build the operator ONCE -------------------------------------------
    T = build_termination_operator(N, k_tc)
    overflow_mask = build_overflow_mask(N)

    # ---- Per-step computation (this part is what you'd run repeatedly) ------
    P = np.ones([N, ])
    P[9] = 4

    PP_flat = np.outer(P, P).reshape(-1)       # vec(PP_mat), length N^2

    D = T @ PP_flat                            # one matvec -> D, shape [N]
    too_high = k_tc * PP_flat[overflow_mask].sum()

    # ---- Cross-check against the explicit loop -----------------------------
    PP_mat = np.outer(P, P)
    D_ref = np.zeros([N, ])
    too_high_ref = 0.0
    for ii in range(N):
        for jj in range(N):
            if ii < jj:
                continue
            if ii + jj + 2 > N:
                too_high_ref += k_tc * PP_mat[ii, jj]
                continue
            D_ref[ii + jj + 1] += k_tc * PP_mat[ii, jj]

    print("D matches loop:        ", np.allclose(D, D_ref))
    print("too_high matches loop: ", np.isclose(too_high, too_high_ref))
    print("operator T: shape", T.shape, "nnz", T.nnz)

    plt.figure(figsize=[5, 3], dpi=200)
    plt.bar(np.arange(1, N + 0.1), D)
    plt.show()
