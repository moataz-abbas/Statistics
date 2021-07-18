# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 06:26:07 2021

@author: mizo_
"""

import stats_v_11 as st
import pandas as pd
import matplotlib.pyplot as plt

d ='C:\Data-Science-with-Python-master\Chapter01\Data\Banking_Marketing.csv'
dx2 = 'C:\Data-Science-with-Python-master\Chapter01\Data\Student_bucketing.csv'
df = pd.read_csv(dx2, header = 0)

print(df.dtypes)

d1 = list(df.marks.dropna())
d2 = [71, 70, 90, 70, 70, 60, 70, 72, 72, 320, 71, 69]
x = sorted (d1)
#print (x)
print (f"len of data = {len(x)}")


a,b,c,d = st.median_iqr(x)
q2 = st.median(x)
print (f"q2 = {q2}")
plt.axhline(q2, color= 'r', linestyle = '--', label = 'median' )



qq1 = x[a:b+1]
# print (qq1)
q1 = st.median(qq1)
print (f"q1 = {q1}")
plt.axhline(q1, color= 'r', linestyle = '--', label = 'q1' )


qq3 = x[c:d+1]
q3 =  st.median(qq3)
print (f"q3 = {q3}")
plt.axhline(q3, color= 'g', linestyle = '--', label = 'q3' )


iqr = q3-q1
print (f"IQR = {iqr}")

lw = q1 - 1.5 * iqr
uw = q3 + 1.5 * iqr

if lw < min(x):
    lw = min(x)
if uw > max(x):
    uw = max(x)
    



print (f"lower whisker = {lw}")
print (f"upper whisker = {uw}")
plt.axhline(lw, color= 'b', linestyle = '--', label = 'lw' )
plt.axhline(uw, color= 'k', linestyle = '--', label = 'uw' )

print(min(x), max(x))

plt.boxplot(x)
plt.legend()
plt.show()