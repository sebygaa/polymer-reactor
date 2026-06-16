import numpy as np
import matplotlib.pyplot as plt 
from scipy. integrate import odeint

k_d = 1E2
k_p = 1E-1

N_max = 400
mat_min_1 = np.diag(np.ones([N_max-1]),-1)
print(mat_min_1) 

def model(y,t):
    I = y[0]
    M = y[1]
    P = y[2:]

    P_min_1 = mat_min_1@P
    #print(P_min_1)
    dIdt = -k_d*I
    lamb0 = np.sum(P)
    # Monomer consumption: radical to P1 & propagation
    dMdt = -2*k_d*I - k_p*M*lamb0
    # General (propagation only)
    dPdt = k_p*M*(P_min_1 - P)
    # P1 from radical to P1
    dPdt[0] = 2*k_d*I -k_p*M*(P[0])
    dydt = np.concatenate([[dIdt, dMdt,], dPdt])
    return dydt

I0 = 1E-2
M0 = 10
y0 = np.zeros([N_max+2,])
y0[0] = I0
y0[1] = M0

y_ret_test = model(y0,0)
print(y_ret_test)
t_dom = np.linspace(0,100,1001)
y_res = odeint(model, y0, t_dom)

P_list = []
plt.figure()
p_index = np.arange(1,N_max+0.5)
for yy,tt in zip(y_res[::100, :], t_dom[::100]): 
    pp = yy[2:]
    P_list.append(pp)
    plt.plot(p_index, pp, 
             label='t = {0:.1f}'.format(tt))

plt.show()
