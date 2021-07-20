# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 06:26:07 2021

@author: mizo_
"""

import stats_v_11 as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


d ='/storage/emulated/0/Python/Data science in python/Data-Science-with-Python-master/Data-Science-with-Python-master/Chapter01/Data/Banking_Marketing.csv'
dx2 = '/storage/emulated/0/Python/Data science in python/Data-Science-with-Python-master/Data-Science-with-Python-master/Chapter01/Data/german_credit_data.csv'
df = pd.read_csv(dx2, header = 0)

#print(df.dtypes)

d1 = list(df.Age.dropna())
d2 = [71, 70, 90, 70, 70, 60, 70, 72, 72, 320, 71, 69]
d3 = [51, 17, 25, 39, 7, 49, 62, 41, 20, 6, 43, 13]
d4 = [ 6, 7, 15, 36, 39, 40, 41, 42, 43, 47, 49]
d5 = [82,76,24,40,67,69,62,75,78,71,32,98,89,78,67,72,82,87,66,56,52]
d6= [1,2,3,4,5,6,7,8]
d7 =[2,4,5,8,10,11,12,14,17,18,21,22,25]
d8= [0, 1, 5, 5, 5, 6, 6, 7, 7, 8, 11, 12, 21, 23, 23, 24]

g = d1
x = sorted (g)
z = pd.DataFrame(g)
print (x)
print (f"len of data = {len(x)}")


def maxi(items, i):
    current = items[0]
    for item in items:
    	if item<=i and item > current:
	            current = item
    return current
    
def mini(items, i):
    items = items[::-1]
    current = items[0]
    for item in items:
    	if item>=i and item < current:
	            current = item
    return current
    
def quarts(q, data):
	n = len(data)
	pos = q * (n-1)
	#a,b,c,d = st.median_iqr_plt(x)
	frac = pos%1
	if pos != 0:
		i = int(pos)
		j = i +1
		quart = data[i] + (data[j] - data[i])*frac
	else:
		quart = data[pos]
		
	return quart
	
#print(f"0.75 = {interpol(.25, x)}")

#a,b,c,d = st.median_divid_plt(x)
# print(x[d])
#print(a,b,c,d)


#qq1 = x[a:b+1]
# print (qq1)
#q1 = st.median(qq1)
q1= quarts(0.25, x)
Q1 = z.quantile(0.25)

#q2 = st.median(x)
q2 = quarts(0.5, x)
Q2 = z.quantile(0.5)

#qq3 = x[c:d+1]
#q3 =  st.median(qq3)
q3 = quarts(0.75, x)
Q3 = z.quantile(0.75)


# IQR
iqr = q3-q1
IQR = Q3-Q1
print(IQR)

# Whiskers
low = q1 - 1.5 * iqr
up = q3 + 1.5 * iqr
uw = maxi(x, up)
lw = mini(x, low )

plt.axhline(max(x), color= 'r', linestyle = '-', label = 'Max' )
plt.axhline(uw, color= 'b', linestyle = '--', label = 'Upper whisker' )
plt.axhline(q3, color= 'g', linestyle = '--', label = 'Q3' )
plt.axhline(q2, color= 'y', linestyle = '--', label = 'Median' )
plt.axhline(q1, color= 'g', linestyle = '--', label = 'Q1' )
plt.axhline(lw, color= 'r', linestyle = '-', label = 'Low whisker' )
plt.axhline(min(x), color= 'b', linestyle = '--', label = 'Min' )

print (f"lower whisker = {lw}")
print (f"q1 = {q1}")   
print(Q1) 
print (f"q2 = {q2}")
print(Q2)
print (f"q3 = {q3}")
print(Q3)
print (f"upper whisker = {uw}")
#print(min(x), max(x))

print (f"IQR = {iqr}")
print(IQR)


plt.boxplot(x, manage_ticks= True)
plt.legend()
plt.show()

"""
boxplot(x, notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, showcaps=None, showbox=None, showfliers=None, boxprops=None, labels=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_ticks=True, autorange=False, zorder=None, *, data=None)[source]
"""