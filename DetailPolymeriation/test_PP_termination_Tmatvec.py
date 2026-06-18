import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# =============================================================================
# Termination-by-combination as a single matrix-vector product:
#
#       D = T @ vec(P_vec @ P_vec.T)
#
# T  : constant [N x N^2] sparse operator (depends only on N and k_tc)
# vec(.) : C-order flatten of the [N, N] outer product PP_mat
#
# Build T once, then reuse it every time P changes.
# =============================================================================


def build_termination_operator(N, k_tc=1.0):
    """Constant [N x N^2] sparse operator T with D = T @ vec(PP_mat).

    C-order flatten => PP_mat[ii, jj] sits at column ii * N + jj.
    Only the lower triangle (ii >= jj) within range (ii + jj + 2 <= N) maps to D;
    the remaining lower-triangle entries are the 'too_high' overflow.
    """
    ii, jj = np.indices((N, N))
    s = ii + jj

    valid = (ii >= jj) & (s + 2 <= N)
    rows = s[valid] + 1                         # target D bin
    cols = ii[valid] * N + jj[valid]            # column in vec(PP_mat)
    data = np.full(rows.shape, k_tc, dtype=float)

    return sp.csr_matrix((data, (rows, cols)), shape=(N, N * N))


if __name__ == "__main__":
    N = 20
    k_tc = 1.0

    # ----- Build the operator ONCE ------------------------------------------
    T = build_termination_operator(N, k_tc)

    # ----- Per-step state ---------------------------------------------------
    P = np.ones([N, ])
    P[9] = 4
    P_vec = P.reshape(-1, 1)                     # [N, 1] column vector

    # ----- The one-liner ----------------------------------------------------
    PP_mat = P_vec @ P_vec.T                     # [N, N] outer product
    D = T @ PP_mat.reshape(-1)                   # D = T @ vec(P_vec @ P_vec.T)

    print("D =", D)
    print("operator T: shape", T.shape, "nnz", T.nnz)

    plt.figure(figsize=[5, 3], dpi=200)
    plt.bar(np.arange(1, N + 0.1), D)
    plt.show()
