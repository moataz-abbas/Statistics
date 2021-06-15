import stats_v_2_1 as st
from math import sqrt


x = [12,9,7,10,11,15,16,8,9,12]
y = 13
#x=[34518, 29540,34889,26764,31429,29962,31084,30506,28546,29560,29304,25852]
#y=31456

print(f"sample: {x}")
print(f"pop mean (Target) value: {y}")

n= len(x)
print(f"N: {n}")

mu= st.mean(x)
print(f"mean: {mu}")

s=st.std(x)
print(f"std: {s}")

sem = st.sem(x,n)
print(f"SEM: {sem}")


t = st.osz_test(x_bar=mu, n=n, u=y, s=s)
print(f"t: {t}")

dof= st.dof(x)
print(f"DOF: {dof}")

md= mu - y
print(f"Mean difference: {md}")


t_score = st.t_table(dof)
print(f"t_score: {t_score}")
print(f"CI of the mean difference: {st.cimd_osz(mu, y, s, -t_score, n)}")

print(f"probability: {st.p_of_t(t, dof, 2)}")
