from math import sqrt, pi, e
from collections import Counter
import numpy as np


def median(lst):
	n=len(lst) # total no. of elements
	mid = int(n/2) # get the mid point
	x = sorted(lst) # sort the lst
	
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


def mean(lst):
	n= len(lst)
	return sum(lst)/n
	
	
def var(lst):
	mu= mean(lst)
	return  sum((x - mu)**2 for x in lst)/(len(lst)-1)


def var_p(lst):
    """ population variance """
    mu= mean(lst)
    return  sum((x - mu)**2 for x in lst)/(len(lst))


def std_p(lst):
    """ population stdev """
    return sqrt(var_p(lst))


def std(lst):
	return sqrt(var(lst))
	
	
def std_error(lst):
	""" 
	Standard of Error:
	using list of array
	"""
	return std(lst)/(sqrt(len(lst)))
	

def dof(lst):
	return len(lst) -1
	

def weighted_mean(lst):
	w = Counter(lst)
	return sum([x*y for x,y in w.items()])/len(lst)

		
def mode(lst):
	if not lst: #To check if lst is empty
		return "The lst is empty!"
	w=Counter(lst)
	_,i = w.most_common(1)[0]
	v = {v for _,v in w.items()} # Set comprehension to get the lst of unique frequencies
	m= [x for x,y in w.items() if y == i] # get the lst of most common elements
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
	

def z_score(x, mu, s):
	""" Z score """
	return (x - mu)/s


def x_of_z(z, mu, s):
	""" x value of a Z score """
	return (z*s) + mu
	
	
def pdf(x, mu, s):
	"""
	Probability distribution function
	"""
	return (1/(s*(sqrt(2 * pi)) ))*(e **(-0.5 *((x-mu)/s)**2))
	

def cdf(a, b, mu=0, s=1, n=10000):
	"""
	Area under normal distribution curve:
		(Z - table calculator)
	# a and b are the start and end values you want to predict the probability of happening
	# mu and s are the mean and std of the normal distribution curve, 
	#     the default values draw a standard model of mean zero and std of 1.
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


def cdf_from_mu(x):
	"""
	CDF of Z score from mean, 
	of standard distribution curve
	"""
	return cdf(0, x, 0, 1, 5000)


def skew(lst):
    """ Measure of skewiness"""
    return (3*(mean(lst)-median(lst)))/std(lst)


def kurt(lst):
    """ Measure of kurtosis """
    m = mean(lst)
    s = std_p(lst)
    ex = [  (z_score(x, m, s))**4 for x in lst]
    return mean(ex)-3
    
def ci(val, mu=0, s=1, n=3000):
	"""
	 Confidence interval:
	 val:  confidence interval in decimals (e.g. 0.95, 0.99)
	 mu: mean ,, s: stdev of the curve
	 returns the range of the mean under normal distrubution curve
	 	
	 """
	p = (1- val)/2
	a =- 3.7
	x = list(np.linspace(a, -1, n)) 
	y = [cdf(a, i, 0, 1, n) for i in x]
	
	r = next(x for x, v in enumerate(y) if v >= p)
	
	res = np.around(np.array(x[r]), 3)
	
	return x_of_z(res,mu,s), x_of_z(-res, mu, s)

def ci_saved(val):
	if val == 0.95:
		return -1.97, 1.96
	elif val == 0.975:
		return -2.237, 2.237
	elif val == 0.99:
		return -2.567, 2.567
	else:
		return "Invalid confidence interval!"

def sem(val,n):
	"""
	Standard Error of the Mean (SEM):
		stdev of all possible means selected from the population.
	s: list of values or stdev of the population
	n: the size of the sample
	"""
	if isinstance(val, list):
		s= std(val)
	else:
		s = val

	return s/sqrt(n)
	

def z_test_1(x_bar, n, u, s):
	"""
	One sample Z-test:
		used to compare mean of population and sample and return z value to be used with z or t tables
	x_bar : mean of the sample
	n : size of the sample
	u : population mean
	s : std of population
		
	"""
	return (x_bar - u)/sem(s,n)


def p_of_z(z):
	""" 
	probability according to z score
	"""
	if z<0:
		a = -3.7
		return round(cdf(a,z,0,1),4)
	else:
		b = 3.7
		return round(cdf(z,b, 0,1),4)
		

def reject_null_h(c, z):
	"""
	Null hypothesis rejected or not
	"""
	a,b = ci_saved(c)
	if z<0:
		r = -z+a
	else:
		r = z-b
		
	if r>=0:
		print("Null hypothesis REJECTED, Significant difference!")
		return 1
	else:
		print("Null hypothesis ACCEPTED, No significant difference!")
		return 0
