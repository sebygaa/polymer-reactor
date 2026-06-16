import numpy as np
import matplotlib.pyplot as plt

N = 20
P = np.linspace(N,1,N)
P = np.ones([N,])
#print(P)
P_vec = np.matrix([P,]).T
print(P_vec)

PP_mat = P_vec*P_vec.T
print(PP_mat)

k_tc = 1

too_high = 0
ind_log = []
D = np.zeros([N,])
for ii in range(N):
    for jj in range(N):
        if ii < jj:
            continue
        if ii + jj + 2 > N:
            too_high += k_tc*PP_mat[ii,jj]
            continue
        D[ii+jj+1] += k_tc*PP_mat[ii,jj]

plt.figure(figsize=[5,3], dpi=200)
plt.bar(np.arange(1,N+0.1), D)

plt.show()
