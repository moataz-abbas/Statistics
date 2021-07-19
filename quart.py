# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 06:26:07 2021

@author: mizo_
"""

import stats_v_11 as st
import pandas as pd
import matplotlib.pyplot as plt

d ='C:\Data-Science-with-Python-master\Chapter01\Data\Banking_Marketing.csv'
dx2 = 'C:/Users/mizo_/Documents/abbasmd.com/workstation/Data science in python/Data-Science-with-Python-master/Data-Science-with-Python-master/Chapter01/german_credit_data.csv'
df = pd.read_csv(dx2, header = 0)

#print(df.dtypes)

d1 = list(df.Age.dropna())
d2 = [71, 70, 90, 70, 70, 60, 70, 72, 72, 320, 71, 69]
d3 = [51, 17, 25, 39, 7, 49, 62, 41, 20, 6, 43, 13]
d4 = [ 6, 7, 15, 36, 39, 40, 41, 42, 43, 47, 49]
#z = pd.DataFrame(d4)


x = sorted (d4)
print (x)
print (f"len of data = {len(x)}")


a,b,c,d = st.median_iqr_plt(x)
q2 = st.median(x)
#Q2 = z.quantile(0.5)
#print(Q2)

plt.axhline(q2, color= 'y', linestyle = '--', label = 'median' )


qq1 = x[a:b+1]
# print (qq1)
q1 = st.median(qq1)
#Q1 = z.quantile(0.25)
# print(Q1)
plt.axhline(q1, color= 'r', linestyle = '--', label = 'q1' )


qq3 = x[c:d+1]
q3 =  st.median(qq3)
#Q3 = z.quantile(0.25)
# print(Q3)
plt.axhline(q3, color= 'g', linestyle = '--', label = 'q3' )


iqr = q3-q1
#IQR = Q3-Q1
#print(IQR)


lw = q1 - 1.5 * iqr
uw = q3 + 1.5 * iqr

if lw < min(x):
    lw = min(x)
if uw > max(x):
    uw = max(x)
    
plt.axhline(lw, color= 'b', linestyle = '--', label = 'lw' )
plt.axhline(uw, color= 'k', linestyle = '--', label = 'uw' )

print (f"lower whisker = {lw}")
print (f"q1 = {q1}")    
print (f"q2 = {q2}")
print (f"q3 = {q3}")
print (f"upper whisker = {uw}")

print (f"IQR = {iqr}")
# print(min(x), max(x))

plt.boxplot(x)
plt.legend()
plt.show()