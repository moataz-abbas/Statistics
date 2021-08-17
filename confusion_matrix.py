# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:01:20 2021

@author: mizo_
"""

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    """ 
    the true results over the total results (percent of true results)
    """
    correct = tp + tn
    total = tp + tn + fn + tn
    return correct/total

def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    """
    Positive Predictive Value: 
    -- How much a positive result weights
    -- The percentage of TRUE (positive) results to all positive 
        results(FALSE AND TRUE)
    
    """
    return tp/(tp + fp)

def recall(tp: int, fp: int, fn: int, tn: int)  -> float:
    """
    Sensitivity:
    -- How much of the True results are picked by the test
    -- Percentage of the True positive to all True results 
        (false negative(the ones not picked by the test) and
         the true positive the true ones picked positive)

    """
    return tp / (tp + fn)

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    
    return 2*p*r/(p+r)
    
    
print (f1_score(70, 4930, 13930, 981070))

