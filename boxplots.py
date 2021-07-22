# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 02:30:06 2021

@author: mizo_
"""
import pandas as pd
import matplotlib.pyplot as plt



def boxplots(d):
    x = sorted (d)
    print(x)
    print (f"Size of data = {len(x)}")
    
    def maxi(items, i):
        """
        Function to find largest data point (datum) in the sample, 
        that is smaller than Q3 + (1.5 x IQR) 
        """
        current = items[0]
        for item in items:
        	if item<=i and item > current:
    	            current = item
        return current
        
    def mini(items, i):
        """
        Function to find the least datum that is 
        bigger than Q1 - (1.5 x IQR)
        """
        items = items[::-1] # Invert the list
        current = items[0]
        for item in items:
        	if item>=i and item < current:
    	            current = item
        return current
        
    def quarts(q, data):
        """
        Linear interpolation to the percentiles and quartiles
        
        """
        n = len(data)
        pos = q * (n-1)
        frac = pos%1
        if pos != 0:
            i = int(pos)
            j = i +1
            quart = data[i] + (data[j] - data[i])*frac
        else:
            quart = data[pos]
    		
        return quart
    	
    q1= quarts(0.25, x)
    q2 = quarts(0.5, x)   
    q3 = quarts(0.75, x)    
    
    # IQR
    iqr = q3-q1
    
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
    
    print (f"Lower whisker = {lw}")
    print (f"Q1 = {q1}")   
    print (f"Q2 = {q2}")
    print (f"Q3 = {q3}")
    print (f"Upper whisker = {uw}")
    print (f"IQR = {iqr}")
    
    plt.boxplot(x, patch_artist= True)
    plt.legend()
    plt.show()
    
    
    
dx2 = 'C:\Data-Science-with-Python-master\Chapter01\Data\german_credit_data.csv'
df = pd.read_csv(dx2, header = 0)

d1 = list(df.Age.dropna())
d2 = [71, 70, 65, 70, 69, 68, 70, 72, 72, 75, 71, 69]
d3 = [51, 17, 25, 39, 7, 49, 62, 41, 20, 6, 43, 13]
d4 = [6, 7, 15, 36, 39, 40, 41, 42, 43, 47, 49, 25]
d5 = [82,76,24,40,67,69,62,75,78,71,32,98,89,78,67,72,82,87,66,56,52]
d6= [1,2,3,4,5,6,7,8]
d7 =[2,4,5,8,10,11,12,14,17,18,21,22,25]
d8= [0, 1, 5, 5, 5, 6, 6, 7, 7, 8, 11, 12, 21, 23, 23, 24]

boxplots(d4)
#print(st.median(d4))