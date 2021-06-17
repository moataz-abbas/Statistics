"""
T test for independent Samples
"""

import stats_v_2_1 as st
#from math import sqrt
import pandas as pd


df= pd.read_csv('/storage/emulated/0/Python/Data science in python/statistics/data/data11_2.csv')

#print(df)

g = list(df['Gender'])
h = list(df['Hands_Up'])
X = list(zip(g,h))
#print(X)

xm = [j for i,j in X if i==1]
print(xm)
xf = [j for i,j in X if i==2]
print(xf)

#print(df)

x1= xm
x2 = xf
#print(x1, x2)

print(f"sample 1: {x1}")
print(f"sample 2: {x2}")

n1= len(x1)
print(f"N1: {n1}")

n2= len(x2)
print(f"N2: {n2}")

mu1= st.mean(x1)
print(f"mean 1: {round(mu1, 3)}")

mu2= st.mean(x2)
print(f"mean 2: {round(mu2, 3)}")


s1=st.std(x1)
print(f"std1: {round(s1,3)}")

s2=st.std(x2)
print(f"std2: {round(s2,3)}")


sem1 = st.sem(x1,n1)
print(f"SEM1: {round(sem1, 3)}")

sem2 = st.sem(x2,n2)
print(f"SEM2: {round(sem2, 3)}")


t = st.t_indep(x1, x2)
print(f"t value: {round(t,3)}")

dof= st.dof(x1) + st.dof(x2)
print(f"DOF: {dof}")

t_score = st.t_table(dof, prop= 0.05, tail=1)
print(f"t crit: {t_score}")


print(f"probability: {st.p_of_t(t, dof, tail=1)}")

md= mu1 - mu2
print(f"Mean difference: {round(md,3)}")



sed = st.sed(x1,x2)
print(f"Standard Error of mean difference: {round(sed,3)}")


cimd1 =st.cimd_2st(x1, x2, t_score)
print(f"CI of Diff: {cimd1}")
