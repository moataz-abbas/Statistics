# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 18:52:39 2021

@author: mizo_
"""

#pylint:disable=W0622
import stats_v_6 as st
from math import sqrt
#import pandas as pd


qm=[76,81,78,76,76,78,76,78,98,88,76,66,44,67,65,59,87,77,79,85,68, 76,77,98,98,99,98,87,67,78]
qpc= [43,33,23,34,31,51,56,43,44,45,32,33,28,39,31,38,21,27,43,46,41,41,48, 56, 56,55,45,68,54,33]

x1 = qm
x2 = qpc
#print(f"sample 1: {x1}")
#print(f"sample 2: {x2}")

#n1= len(x1)
#print(f"N1: {n1}")

#n2= len(x2)
#print(f"N2: {n2}")

#mu1= st.mean(x1)
#print(f"mean 1: {round(mu1, 3)}")

#mu2= st.mean(x2)
#print(f"mean 2: {round(mu2, 3)}")


#s1=st.std(x1)
#print(f"std1: {round(s1,3)}")

#s2=st.std(x2)
#print(f"std2: {round(s2,3)}")


#sem1 = st.sem(x1,n1)
#print(f"SEM1: {round(sem1, 3)}")

#sem2 = st.sem(x2,n2)
#print(f"SEM2: {round(sem2, 3)}")

#def src(r,n):
#	return sqrt((1- (r**2))/(n-2))


dof= st.dof_c(x1)
print(f"DOF: {dof}")


r= st.r_xy(x1, x2)
print(f"pearson r = {r}")

#sr= src(rxy, n1)
#print(f"sr = {sr}")

# def r_crit(**vars):
# 	""" values of Correlation coefficient needed for rejection of null hypothesis (critical value of r)
# 	degrees of freedom = n of the pairs - 2
# 	p = risk of significance, default 0.05
# 	tails = tails of the distribution, default (2)
# 	"""
# 	df = vars.get('df', 1)
# 	p = vars.get('p', 0.05)
# 	tail = vars.get('tail', 2)
# 	#print(df, p, tail)
# 	t= st.t_table(**vars)
# 	return sqrt((t**2)/(t**2 + df))

dof=25

rc = st.r_crit(df = dof)
print(f"r crit: {rc}")


#def reject_null_h_f(func, d1, d2, p=0.05, a=1000, n=10**6 ):
	
def reject_null_h_all(val, func, **vars):
	"""
	Universal Null hypothesis rejected or not
	- val: the obtained value of significance representinh the research hypothesis
	funtion returning the critical value of rejection for a given dof(s), p value (risk of significance), and no. of tails as **kwargs arguments
	"""
	print(vars)
	crit = func(**vars)
		
	if abs(val) > crit:
		return f"Null hypothysis rejected! \nas Value obtained = {round(val,4)}, \nand Critical value= {round(crit,4)}."
	else:
		return f"Null hypothysis accepted. \nas Value obtained = {round(val,4)}, \nand Critical value = {round(crit,4)}."



print(st.reject_null_h_all(val=r, func=st.r_crit, df= dof, p= 0.05, tail= 2))




#md= mu1 - mu2
#print(f"Mean difference: {round(md,3)}")



#sed = st.sed(x1,x2)
#print(f"Standard Error of mean difference: {round(sed,3)}")


#cimd1 =st.cimd_2st(x1, x2, t_crit)
#print(f"CI of Diff: {cimd1}")
