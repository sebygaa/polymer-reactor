import numpy as np
import matplotlib.pyplot as plt 
from scipy. integrate import odeint

k_d = 1E3
k_p = 1.5
k_trm = 8E-3
k_trp = 1E-1

N_max = 400
mat_min_1 = np.diag(np.ones([N_max-1]),-1)
#print(mat_min_1) 
mat_1toN = np.arange(1, N_max+0.1)

def model(y,t):
    I = y[0]
    M = y[1]
    P = y[2:2+N_max]
    D = y[2+N_max:]
    lamb0 = np.sum(P)
    #mu0 = np.sum(D)
    mu1 = np.sum(mat_1toN*D)
    # Chain transfer to monomer reaction
    r_trm = k_trm*M*P # r_trm is a vector with size of [N_max,]
    r_trm_sum = np.sum(r_trm)
    
    r_trp_P = k_trp*P*mu1            # Consuming P_i with all dead polymers
    r_trp_D = k_trp*lamb0*D*mat_1toN # Consuming D_i with all active polymers

    P_min_1 = mat_min_1@P # P_(i-1) {=polymer i-1} P_min_1 is also a vector with size of [N_max,]
    #print(P_min_1)
    dIdt = -k_d*I
    # Monomer consumption: radical to P1 & propagation
    dMdt = -2*k_d*I - k_p*M*lamb0 -r_trm_sum
    
    # Active polymer
    # General (propagation - chain transfer to monomer)
    dPdt = k_p*M*(P_min_1 - P) - r_trm - r_trp_P + r_trp_D
    # P1 from radical to P1
    dPdt[0] += 2*k_d*I -k_p*M*(P[0])+ r_trm_sum - r_trp_P[0] + r_trp_D[0]
    
    # Dead polymer
    dDdt = r_trm + r_trp_P - r_trp_D
    
    dydt = np.concatenate([[dIdt, dMdt,], dPdt, dDdt])
    return dydt

I0 = 5E-2
M0 = 10
y0 = np.zeros([2*N_max+2,])
y0[0] = I0
y0[1] = M0

y_ret_test = model(y0,0)
#print(y_ret_test)
t_dom = np.linspace(0,100,1001)
y_res = odeint(model, y0, t_dom)

P_list = []
D_list = []
plt.figure(figsize = [5,4.2], dpi=200)
p_index = np.arange(1,N_max+0.5)
for yy,tt in zip(y_res[::100,:], t_dom[::100]): 
    pp = yy[2:N_max+2]
    dd = yy[N_max+2 : 2*N_max+2]
    P_list.append(pp)
    D_list.append(dd)
    plt.plot(p_index, pp, 
             label='t = {0:.1f}'.format(tt))
plt.xlabel('Mw_polymer / Mw_monomer')
plt.ylabel('Active Polymer Concen. (mol/m$^{3}$)')
plt.legend()
plt.tight_layout()
#plt.show()

print('D_list = ', len(D_list))

plt.figure(figsize = [5,4.2], dpi=200)
for dd,tt in zip(D_list, t_dom[::100]):
    plt.plot(p_index, dd,
             label='t= {0:.1f}'.format(tt))
plt.xlabel('Mw_polymer / Mw_monomer')
plt.ylabel('Dead Polymer Concen. (mol/m$^{3}$)')
plt.legend()
plt.tight_layout()

# Overall polymer
P_res_mat = np.matrix(P_list)
D_res_mat = np.matrix(D_list)

PD_res = P_res_mat + D_res_mat

plt.figure(figsize = [5,4.2], dpi=200)
for pdpd, tt in zip(PD_res,t_dom[::100]):
    plt.plot(p_index, np.array(pdpd.T[:,0]),
             label='t= {0:.1f}'.format(tt))
plt.xlabel('Mw_polymer / Mw_monomer')
plt.ylabel('Dead Polymer Concen. (mol/m$^{3}$)')
plt.legend()
plt.tight_layout()
plt.show()


