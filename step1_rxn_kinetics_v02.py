# %%
# Importing Packages
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# %%
# Defning Key Variables
# %%
kd_test = 0.8
kp_test = 15
ktc_test = 0.02
ktd_test = 0.02
ktrm_test = 15E-3
ktrp_test = 10E-3
# kp >> ktrm > kd >> ktc, ktd, ktrp 순서여야함
# 만약 kp = 5이고 ktrm = ktrp = 1 m³/(mol·s) = 1000 L/(mol·s) 이라면 비현실적...
# → kp와 같은 수준으로 너무 높음 (실제로는 kp의 1/1000 ~ 1/100 수준)
f_test = 0.8 # Initiator efficiency (80% assumed)

par_list = [kd_test, kp_test, ktc_test, ktd_test, 
            ktrm_test, ktrp_test, f_test,]

   
Tc_const = 200+273 # K
dH_p = -100*1000 # 100 kJ/mol = 100,000 J/mol
Mw_mono = 28.05# g/mol

# Dnesity (rho)
P_assu = 3000*1E5 # 3000 bar (1E8 Pa) & 150+273 K
T_assu = 150+273  # K
R_gas = 8.3145    # J/mol/K
C_assu = P_assu*R_gas*T_assu # mol/m^3
rho = Mw_mono*C_assu # g/m^3
print('rho =')
print(rho)

# Heat capacity Cp
Cp_C2 = 42.9  # J/mol/K
Cp_ass = Cp_C2/Mw_mono  # J/g/K

# Overall heat transfer coefficients (U)
U_heat = 400    # J/s/m^2/K

# Diameter
D = 0.05       # 10 cm = 0.10 m 
# %%
# Form of ODE equations

def rxn1(y,t,arg_list):
    lamb0 = y[0]
    lamb1 = y[1]
    lamb2 = y[2]
    mu0 = y[3]
    mu1 = y[4]
    mu2 = y[5]
    ini = y[6]
    mono = y[7]
    T = y[8]
    kd, kp, ktc, ktd, ktrm, ktrp, f = arg_list

    # mu3 from Schulz-Zimm distribution (gamma distribution assumptio)
    # Hulburt & Katz approximation
    mu3_guess = mu2*(2*mu0*mu2 - mu1**2)/(mu0+1E-6)/(mu1+1E-6)
    mu3 = max(mu3_guess, 0)
    
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
    dmu2_dt += ktrp*(lamb2*mu1 - lamb0*mu3)

    # Energy balance (T)
    dT_dt = (-dH_p)/rho/Cp_ass*kp*mono*lamb0 - 4*U_heat/rho/Cp_ass/D*(T-Tc_const)

    # Overall -> return
    dydt_list = [dlamb0_dt, dlamb1_dt, dlamb2_dt,
                 dmu0_dt, dmu1_dt, dmu2_dt,
                 dini_dt, dmono_dt, dT_dt]
    dy_dt = np.array(dydt_list)
    return dy_dt

# %%
# Initial conditions
# %%
mono_0 = 2.005E4 # mol/L at 1000 bar 600 K
ini_0 = 100.0
T_0 = 150+273   # K
y0 = np.zeros([9,])
y0[6] = mono_0
y0[7] = ini_0
y0[8] = T_0
# %%
t_ran = np.arange(0,20.01, 0.02)
y_res = odeint(rxn1,y0,t_ran, args = (par_list,),)

# %%
# Molecular weights: Mn, Mw, PDI
# Mn: 수평균분자량 number-average Molecular weight)
lamb0_res = y_res[:,0]
lamb1_res = y_res[:,1]
lamb2_res = y_res[:,2]
mu0_res = y_res[:,3]
mu1_res = y_res[:,4]
mu2_res = y_res[:,5]
ini_res = y_res[:,6]
mono_res = y_res[:,7]
T_res = y_res[:,8]

Mn_res = Mw_mono*(lamb1_res+mu1_res)/(lamb0_res+mu0_res)
Mw_res = Mw_mono*(lamb2_res+mu2_res)/(lamb1_res+mu1_res)
PDI_res = Mw_res/Mn_res
# %%
# Graph
# %%
plt.figure(figsize=[5/1.6, 3.8/1.6], dpi=300)
plt.plot(t_ran, Mn_res, 
         label = 'Mn (g/mol)')
plt.legend(fontsize= 9)

plt.figure(figsize=[5/1.6, 3.8/1.6], dpi=300)
plt.plot(t_ran, Mw_res, 
         label = 'Mw (g/mol)')
plt.ylabel('molecular weight (g/mol)')
plt.legend(fontsize= 9)

plt.figure(figsize=[5/1.6, 3.8/1.6], dpi=300)
plt.plot(t_ran, T_res, 
         label = 'T (K)')
plt.ylabel('temperature (K)')
plt.legend(fontsize= 9)

plt.show()

# %%
#plt.plot(t_ran, y_res[:,-2])
#plt.show()
# %%
