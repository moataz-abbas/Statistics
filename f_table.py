import stats_v_2_2 as st
import numpy as np
#importmatplotlib.pyplot as plt
from math import sqrt, gamma


def beta(x,y):
	"""Beta funcrion"""
	return (gamma(x)*gamma(y))/gamma(x+y)
	
def f_pdf(x, d1, d2):
    a,b = d1/2, d2/2
    g1= ((d1*x)**d1)*(d2**d2)
    # print(f"g1:{g1}")
    g2= pow((d1*x + d2),(d1+d2))
    # print(f"g2:{g2}")
    rv = beta(a,b)
    # print(f"rv:{rv}")
    g3= x* rv
    # print(f"g3:{g3}")
    return (sqrt(g1/g2))/g3




def f_table(d1, d2, p=0.05, a=7, n=10**6):
    """
    F-distribution table
    Parameters
    ----------
    d1 : float
        degrees of freedom df1.
    d2 : float
        degrees of freedom df2.
    p : TYPE, optional
        probability. The default is 0.05.
    a : TYPE, optional
        the end point of the skewed tail. The default is 7.
    n : TYPE, optional
        fractions of the area under the curve(dx). The default is 100000.

    Returns
    -------
    res : float
        F critical value.

    """
    
    x = list(np.linspace(a, 10**-5, n))
    #print(x)
    dx = a/n
    #print(f"dx: {a/n}")
    y=0 #sum of the area
    c=0    # counter of dx
    for i in x:
        y+= dx * f_pdf(i, d1, d2)
        # print(y)
        if y >= p:
            break
        c+=1
    res = a - (c * dx)
    return res

def f_p(d1, d2, p=1.0, a=7, n = 10**6):
    """ calculate the extent of the the right end at with area under the curve = 1.0"""
    
    x = list(np.linspace(10**-2, a, n))
    #print(x)
    dx = a/n
    #print(f"dx: {a/n}")
    y=0 #sum of the area
    c=0    # counter of dx
    for i in x:
        y+= dx * f_pdf(i, d1, d2)
        z = round(y,5)
        #print(y,z)
        if z >= p:
            print(z)
            break
        c+=1
    res = (c * dx)
    return res



d1= 6
d2= 6
p= 0.05
a= 10**6


#x = np.linspace(10**-3, 3, 1000)
s=f_p(d1, d2, p=1, a=10000, n=10**6)
print(s)
f = f_table(d1, d2, p=p, a= s)
print(f)


#y=[st.f_pdf(i, d1, d2) for i in x]
# z= [f_cdf(10**-10, i, d1, d2) for i in x]
# z1 = f_cdf(10**-10, 10, d1, d2)
# print(z1)
#plt.plot(x,y)
#plt.show()
