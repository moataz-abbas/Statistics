# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:01:20 2021

@author: mizo_

####Confusion Matrix###
								
_Predicted_  ||_____Actual values:___|
 _values:____||_Positive _|_Negative_|
_Positive ___||___TP ____|____FP____|
_Negative __||____FN ___|____TN ___|
"""

def accuracy(tp, fp, fn, tn):
    """ 
    the true results over the total results (percent of true results)
    => It is used if the data is balanced
    """
    correct = tp + tn
    total = tp + tn + fn + tn
    return correct/total

def precision(tp, fp, fn, tn):
    """
    Precision (Positive Predictive Value): 
    <<Predicted True positive over all predicted positive>>
    -- How much a positive result weights
    -- Correctly predicted as positive out of all positive predictes values.
    *** The percentage of TRUE positive results to all PREDICTED positive results(FALSE AND TRUE) - (a measure in the predicted positive lane)
    == If this value is higher tan recall, that means the test is very selective of the prediction, highly specific, and most probably of low sensitivty. i.e. the test is more likely to miss on positive cases (high false negative). 
    and less likely to give false positive (low false positive)
   => high specificity, low sensitivity.
   >> Low FP, High FN
    
    """
    return tp/(tp + fp)

def recall(tp, fp, fn, tn):
    """
    Sensitivity/Recall/True positive rate(TPR):
    <<Predicted True positive over all actual positive>>
    -- How much of the True results are picked by the test as positive 
    -- Out of all the actual positives, how much are predicted as true correctly
    -- Percentage of the True positive to all True results (false negative(the ones not picked by the test) and the true positive the true ones picked positive)
    *** A measure in the Actual positive lane, TP over all the Actual positive 
    =>If recall is higher than precision, the test is very sensitive, and of low selction criteria. i.e. the false negative are low and false positive will be higher. The test tends to label more subjects as positive with higher tendency for wrong positive
    >> High sensitivity and low specificty
    >> High FP, Low FN
    If the impact of FN is much higer than false positive 
    """
    return tp / (tp + fn)
 
def specificity(tp, fp, fn, tn):
	"""
	Correctly predicted negative out of all
	actual negative values
	<<Predicted True negative by test over all actual negative>>
	"""
	return tn/(fp+tn)

def error1(tp, fp, fn, tn):
	""" 
	Type1 Error (False positive rate):
	- FP (wrongly predicted positive) over All actual negative values
	(inverse specificity!!)
	"""
	return fp/(fp+tn)

def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    
    return 2*p*r/(p+r)
    
def f_beta(tp, fp, fn, tn, beta):
    """
    F beta score:
	if both FP and FN are important use beta=1
    if minimizing FP is important decrease beta value,
    if mini FN is the goal increase beta
    """
    p=precision(tp, fp, fn, tn)
    r=recall(tp,fp,fn,tn)
    return ((1+beta)*2*p*r)/((beta*2*r)+p)
