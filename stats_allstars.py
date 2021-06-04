#pylint:disable=C0116
#pylint:disable=W0622
#pylint:disable=C0103
#pylint:disable=R1705
from math import sqrt, pi, e
from collections import Counter
import numpy as np


def median(list):
	n=len(list) # total no. of elements
	mid = int(n/2) # get the mid point
	x = sorted(list) # sort the list
	
	def even(lst_e, n):
		""" function to deal with even series"""
		return (lst_e[n - 1] + lst_e[n])/2
	
	def odd(lst_o, n):
		""" function to deal with odd series"""
		return lst_o[n]
		
	if n%2 == 0: #if clause to check if series is even or odd
		return even(x, mid)
	else:
		return odd(x, mid)

def mean(list):
	n= len(list)
	return sum(list)/n
	
def var(list):
	mu= mean(list)
	return  sum((x - mu)**2 for x in list)/(len(list)-1)

def std(list):
	return sqrt(var(list))
	
def std_error(list):
	return std(list)/(sqrt(len(list)))

def dof(list):
	return len(list) -1

def weighted_mean(lst):
	w = Counter(lst)
	return sum([x*y for x,y in w.items()])/len(lst)
	
def mode(lst):
	if not lst: #To check if list is empty
		return "The list is empty!"
	w=Counter(lst)
	_,i = w.most_common(1)[0]
	v = {v for _,v in w.items()} # Set comprehension to get the list of unique frequencies
	m= [x for x,y in w.items() if y == i] # get the list of most common elements
	if len(v) > 1: # check that we have a mode, and display it according to its number
		return m
	else:
		return 0
		
		
def r_xy(x,y):
	""" Correlation coefficient"""
	x2 = [i**2 for i in x]
	y2 = [i**2 for i in y]
	xy = [x*y for x,y in zip(x,y)]
	ex = sum(x)
	ey = sum(y)
	exy = sum(xy)
	ex2 = sum(x2)
	ey2 = sum(y2)
	n = len(x)
	return (n*exy - ex*ey)/(sqrt((n*ex2 - ex**2)*(n*ey2 - ey**2)))
	

def z_score(x,lst):
	""" Z score """
	return (x - mean(lst))/std(lst)
	
	
def pdf(x, mu, s):
	"""
	Probability distribution function
	"""
	return (1/(s*(sqrt(2 * pi)) ))*(e **(-0.5 *((x-mu)/s)**2))
	

def cdf(a, b, mu=0, s=1, n=10000):
	"""
	# a and b are the start and end values you want to predict the probability of happening
	# mu and s are the mean and std of the normal distribution curve, the default values draw a standard model of mean zero and std of 1.
	# n reflects the density of the plot to increase the accuracy of the results
	"""
	dx= (b-a)/n
	ab= np.linspace(a,b,n)
	Efxk=0
	for x in ab[1:-2]:
			Efxk += pdf(x, mu, s)
		
	outers = (pdf(ab[0], mu,s) 
					+ pdf(ab[-1], mu, s))/2
	
	return dx * ( Efxk + outers)


def skew(lst):
	return (3*(mean(lst)-median(lst)))/std(lst)

