# %%
# Importing Packages
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# %%
# Defning Key Variables
# %%
kd_test = 10
kp_test = 5
ktc_test = 1
ktd_test = 1
ktrm_test = 1
ktrp_test = 1

f_test = 0.8 # Initiator efficiency (80% assumed)

par_list = [kd_test, kp_test, ktc_test, ktd_test, 
            ktrm_test, ktrp_test, f_test,]
            

# %%
# Form of ODE equations
#mu3 = 200
def rxn1(y,t,arg_list):
    lamb0 = y[0]
    lamb1 = y[1]
    lamb2 = y[2]
    mu0 = y[3]
    mu1 = y[4]
    mu2 = y[5]
    ini = y[6]
    mono = y[7]
    print(arg_list)
    kd, kp, ktc, ktd, ktrm, ktrp, f = arg_list

    # mu3 from Schulz-Zimm distribution (gamma distribution assumptio)
    # Hulburt & Katz approximation
    mu3_guess = mu2*(2*mu0*mu2 - mu1**2)/(mu0+1E-6)/(mu1+1E-6)
    mu3 = min(mu3_guess, 5000)
    
    # Initiator and monomor consumption
    dini_dt = -kd*ini
    dmono_dt = -kp*mono*lamb0
    
    # Growth radical moment (0th, 1st, 2nd)
    # lamb0
    dlamb0_dt = 2*f*kd*ini - (ktc+ktd)*lamb0**2
    # lamb1
    term1_1 = kp*mono*lamb0
    term1_2 = ktrm*mono*(lamb0-lamb1)
    term1_3 = ktrp*(lamb0*mu2 - lamb1*mu1)
    term1_4 = -(ktc+ktd)*lamb0*lamb1
    dlamb1_dt =  term1_1 + term1_2 + term1_3+term1_4
    # lamb2
    term2_1 = kp*mono*(2*lamb1+lamb0)
    term2_2 = ktrm*mono*(lamb0 - lamb2)
    term2_3 = ktrp*(lamb0*mu3 - lamb2*mu1)
    term2_4 = -(ktc+ktd)*lamb0*lamb2
    dlamb2_dt = term2_1 + term2_2 + term2_3 + term2_4

    # Growth of dead polymer moment
    # mu0
    term3_1 = ktrm*mono*lamb0
    term3_2 = (0.5*ktc+ktd)*lamb0**2
    # term3_3 = -ktrp*lamb0*mu1 # Claude 가 찾은 오류
    dmu0_dt = term3_1 + term3_2 
    # mu1
    dmu1_dt = ktrm*mono*lamb1 + (ktc+ktd)*lamb0*lamb1
    dmu1_dt += ktrp*(lamb1*mu1 - lamb0*mu2) # Claude가 찾은 오류
    # mu2
    term4_1 = ktrm*mono*lamb2
    term4_2 = ktd*lamb0*lamb2
    term4_3 = ktc*(lamb0*lamb2+lamb1**2)
    dmu2_dt = term4_1 +term4_2+term4_3
    dmu2_dt += ktrp*(mu1*lamb2 - lamb0*mu3)

    dydt_list = [dlamb0_dt, dlamb1_dt, dlamb2_dt,
                 dmu0_dt, dmu1_dt, dmu2_dt,
                 dini_dt, dmono_dt]
    dy_dt = np.array(dydt_list)
    return dy_dt

# %%
# Initial conditions
# %%
mono_0 = 0.30 # mol/L at 20 bar 800 K
ini_0 = 0.01
y0 = np.zeros([8,])
y0[-1] = mono_0
y0[-2] = ini_0
# %%
t_ran = np.arange(0,200.01, 0.2)
y_res = odeint(rxn1,y0,t_ran, args = (par_list,),)

# %%
# Graph

# %%
plt.plot(t_ran, y_res[:,0])
# %%
plt.plot(t_ran, y_res[:,-1])
# %%
