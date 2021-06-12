import stats_allstars as st
from math import sqrt

#z = st.z_test_1(100, 36, 99, 2.5 )
#print(0.5-st.cdf_from_mu(z))

#print(st.ci(val= .975, n=2000))
#print(st.p_of_z(t))

#print(st.reject_null_h(0.99, 1.96))
#print(st.ci(.975),st.ci(.99))
#print(st.ci_saved(0.99))

x = [12,9,7,10,11,15,16,8,9,12]

n= len(x)
print(f"N: {n}")

mu= st.mean(x)
print(f"mean: {mu}")

s=st.std(x)
print(f"std: {s}")

sem = st.sem(x,n)
print(f"SEM: {sem}")

t = st.z_test_1(x_bar=13, n=n, u=mu, s=s)
print(f"T: {t}")



