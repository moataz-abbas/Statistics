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
	"""sample variance (n-1)"""
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
	"""sample stdev"""
	return sqrt(var(lst))
	
	
def std_error(lst):
	""" 
	Standard of Error:
	using a data list
	"""
	return std(lst)/(sqrt(len(lst)))
	

def dof(val):
	"""
	Degrees of freedom
	no. of elements in a sample - 1
	either provide the sample in alist format or the number of elements as a single number
	"""
	if isinstance(val, list):
		r= len(val)
	else:
		r = val
	
	return r -1
	

def weighted_mean(lst):
	w = Counter(lst)
	return sum([x*y for x,y in w.items()])/len(lst)

		
def mode(lst):
	if not lst: #To check if lst is empty
		return "The lst is empty!"
	w=Counter(lst)
	_,i = w.most_common(1)[0] # assign the most common frequency to variable 'i'
	v = {v for _,v in w.items()} # Set comprehension to get a list of unique frequencies
	m= [x for x,y in w.items() if y == i] # get the list of most common elements that occured 'i' times in the data set
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
	
	
def pdf(x, mu=0, s=1):
	"""
	Probability distribution function
	if used without re-assigning mu(mean =0),
	 and s(stdev = 1), it returns the standard 
	 normal distribution curve
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

def ci2(val, mu=0, s=1, n=10000):
	"""
	 Confidence interval:
	 val:  confidence interval in decimals (e.g. 0.95, 0.99)
	 mu: mean ,, s: stdev of the curve
	 returns the range of the mean under normal distrubution curve
	 	
	 """
	p = (1- val)/2
	p=-p
	#print(f"p: {p}")
	#a =- 3.648
	a=-3.6 * s
	x = list(np.linspace(a, mu, n)) 
	dx = a/n
	#print(f"dx: {dx}")
	#y = [st.cdf(a, i, 0, 1, n) for i in x]
	y=0
	c=0
	for i in x:
		#print(f"i: {i}")
		y+= dx * pdf(i, mu, s)
		#print(f"y before break: {y}")
		if y <= p:
			#print(f"y in if: {y}")
			break
		c+=1
		#print(f"c: {c}")
		
	result = round(a - (c * dx), 2)
	
	return result, - result

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
	

def osz_test(x_bar, n, u, s):
	"""
	One sample Z-test:
		used to compare mean of population and sample and return z value to be used with z or t tables
	x_bar : mean of the sample
	n : size of the sample
	s : std of sample
	u : population mean (or given as "Test value")
		
	"""
	return (x_bar - u)/sem(s,n)


def p_of_z(z):
	""" 
	probability from periphery of curve tails,
	according to z score, either from left or right of a standard normal distribution curve
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
	p= (1-c)/2
	if r>=0:
		print("Null hypothesis REJECTED, \nStatistically Significant difference!"
		+f"\np<{round(p, 3)}")
		return 1
	else:
		print(f"Null hypothesis ACCEPTED, \nNo Statistically significant difference!\np>{round(p, 3)}")
		return 0
