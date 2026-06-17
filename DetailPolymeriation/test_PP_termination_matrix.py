import numpy as np
import matplotlib.pyplot as plt

N = 20
P = np.linspace(N, 1, N)
P = np.ones([N, ])
# print(P)
P_vec = P.reshape(-1, 1)          # (N, 1) column vector
print(P_vec)

PP_mat = P_vec @ P_vec.T          # (N, N) outer product
print(PP_mat)

k_tc = 1

# --- Vectorized version of the double loop -----------------------------------
# Index grids: ii = row index, jj = column index
ii, jj = np.indices((N, N))

# Only the lower triangle (ii >= jj) is used, matching `if ii < jj: continue`.
lower = ii >= jj

# Sum of indices (ii + jj). The dead-chain bin is ii + jj + 1.
s = ii + jj

# Contributions that overflow the available chain lengths: ii + jj + 2 > N
overflow = lower & (s + 2 > N)
too_high = k_tc * PP_mat[overflow].sum()

# Valid contributions go into D[ii + jj + 1]
valid = lower & (s + 2 <= N)
bins = (s[valid] + 1)             # target indices into D
weights = k_tc * PP_mat[valid]

D = np.bincount(bins, weights=weights, minlength=N)[:N]

plt.figure(figsize=[5, 3], dpi=200)
plt.bar(np.arange(1, N + 0.1), D)

plt.show()
