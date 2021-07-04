# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 15:44:53 2021

@author: mizo_
"""

import stats_v_9 as st

# x=[212, 147, 103, 50, 46, 42]

# chig = st.chi_gof(x)
# print(f"GOF: {chig}")

# dof = st.chi_dof(x)
# print(f"dof: {dof}")

# chi_crit = st.chi_crit(df=dof, p =0.05, tail=1)
# print(f"chi crit = {chi_crit}")

def chi_gof(**vars):
    """
    chi goodness of fit obtained value (X^2):
    =========================================
    accepts arguments as a list of the observed results (ob)
    and a list of expected results to compair (ex).
    or only observed results and the expected result will calculated,
    as equally divided proportions among the observed values 
    
    """
    ob = vars.get('ob')   #observed values
    ex = vars.get('ex', float(sum(ob)/len(ob))) #expected values
    
    
    if isinstance(ex, float):
        #convert the calculated expected vallue into 
        #a list to fit the sum below
        n = len(ob)
        ex = [ex]*n
    
    X2 = sum([((o - e)**2)/e for o,e in zip(ob,ex)])
    return X2


      
ox = [48,35,15,3] #observed
ex = [58,34.5,7,0.5] #expected

gof= chi_gof(ob= ox, ex=ex )
dof = st.chi_dof(ox)
print(f"dof: {dof}")
print(gof)
chi_crit = st.chi_crit(df=dof, p =0.05, tail=1)[1]
print(f"chi crit = {chi_crit}")

