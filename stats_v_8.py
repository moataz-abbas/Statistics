from math import sqrt, pi, e, gamma
#from scipy.stats import beta
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
	

def z_score(x, mu, s):
	""" Z score """
	return (x - mu)/s


def x_of_z(z, mu, s):
	""" x value of a Z score """
	return (z*s) + mu
	
	
def n_pdf(x, mu=0, s=1):
	"""
	Probability distribution function
	if used without re-assigning mu(mean =0),
	 and s(stdev = 1), it returns the standard 
	 normal distribution curve
	"""
	return (1/(s*(sqrt(2 * pi)) ))*(e **(-0.5 *((x-mu)/s)**2))
	

def n_cdf(a, b, mu=0, s=1, n=10000):
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
			Efxk += n_pdf(x, mu, s)
		
	outers = (n_pdf(ab[0], mu,s) 
					+ n_pdf(ab[-1], mu, s))/2
	
	return dx * ( Efxk + outers)


def cdf_from_mu(x):
	"""
	CDF of Z score from mean, 
	of standard distribution curve
	"""
	return n_cdf(0, x, 0, 1, 5000)


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
	"""*** first edition,, very slow***
	 Confidence interval:
	 val:  confidence interval in decimals (e.g. 0.95, 0.99)
	 mu: mean ,, s: stdev of the curve
	 returns the range of the mean under normal distrubution curve
	 	
	 """
	p = (1- val)/2
	a =- 3.7
	x = list(np.linspace(a, -1, n)) 
	y = [n_cdf(a, i, 0, 1, n) for i in x]
	
	r = next(x for x, v in enumerate(y) if v >= p)
	
	res = np.around(np.array(x[r]), 3)
	
	return x_of_z(res,mu,s), x_of_z(-res, mu, s)

def ci2(val, mu=0, s=1, n=10000):
	"""
	 Confidence interval: 2 tailed
	 val:  confidence interval in decimals (e.g. 0.95, 0.99)
	 mu: mean ,, s: stdev of the curve
	 returns the range of the mean under normal distrubution curve
	 	
	 """
	p = -(1- val)/2
	a=-3.6
	x = list(np.linspace(a, 0, n)) 
	dx = a/n
	y=0
	c=0
	for i in x:
		y+= dx * n_pdf(i)
		if y <= p:
			break
		c+=1
		
	res = round(a - (c * dx), 3)
	
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
		measures how far the sample mean (average) of the data is likely to be from the true population mean. The SEM is always smaller than the SD.
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
		return round(n_cdf(a,z,0,1),4)
	else:
		b = 3.7
		return round(n_cdf(z,b, 0,1),4)
		

def reject_null_h(c, z):
	"""
	Null hypothesis rejected or not
	"""
	a,b = ci2(c)
	if z<0:
		r = -z+a
	else:
		r = z-b
	p= (1-c)/2
	if r>=0:
		print("Null hypothesis REJECTED, \nStatistically Significant difference!"
		+f"\np<{round(p, 3)}")
		return r
	else:
		print(f"Null hypothesis ACCEPTED, \nNo Statistically significant difference!\np>{round(p, 3)}")
		return r
		
		
def coh_d(x_bar, u, s):
	"""Cohen's d for the effect size for one sample Z-test
	x_bar: sample mean
	u: population mean
	s: population stdev
	0 to 0.2 --> a small effect 
	0.2 to 0.5 --> a medium sized effect
	> 0.5 --> a large effect size
	"""
	return (x_bar - u)/s
	

def es(mu1, mu2, v1, v2):
	"""
	Effect size: extension to Cohen's d for 2 samples
	mu1 and 2: means of samples 1 and 2
	v1 and v2 : variance of samples 1 and 2
	"""
	return (mu1-mu2)/(sqrt((v1 + v2)/2))


def cimd_osz(x_bar, u, s, t, n):
	"""confidence interval of mean difference"""
	md = (x_bar - u)
	r = t * sem(s, n)
	
	if md-r < md+r:
		return md-r, md+r
	else:
		return md+r, md-r
		
######T TEST 2 SAMPLES#####
	

def pdf_t(x, v):
	""" Probability distribution function of T tables"""
	return ((gamma((v+1)/2))/(sqrt(v*pi)* gamma(v/2))) * ((1+ (x**2/v))**(-1*(v +1)/2))


def t_table(**vars):
	"""
	 T* - T table, critical t value
	 Confidence interval: 2 tailed
	 val:  probaility relayed to level of risk or level of significance (e.g. 0.05, 0.025)
	 df: degrees of freedom
	tail: 1= 1 tail, 2= 2 tailed
	 	
	 """
	df = vars.get('df')
	p = vars.get('p', 0.05)
	tail = vars.get('tail', 2)
	n= vars.get('n', 10000)
	
	p = -(p)/tail
	if df == 1:
		a= -770
	elif df == 2:
		a = -80
	elif df <= 5:
		a = -25
	else:	
		a= -6
	x = list(np.linspace(a, 0, n)) 
	dx = a/n
	y=0
	c=0
	for i in x:
		y+= dx * pdf_t(i, df)
		if y <= p:
			break
		c+=1
		
	res = round(a - (c * dx), 3)
	return -res


def p_of_t(t, df, tail=2, p = 0.05, n=10000):
	"""
	 Probability of the null hypothesis being true,, 
	 null hypothysis will be rejected if p<0.05
	 
	 t: t value
	 df: degrees of freedom
	 p: risk 
	 	
	 """
	if df == 1:
		a= -770
	elif df == 2:
		a = -80
	elif df <= 5:
		a = -25
	else:	
		a= -6
		
	if t<0:
		r = a - t
	else:
		a = -a
		r = a - t
	
	x = list(np.linspace(t, a, n))
	dx = r/n
	Efxk=0
	for i in x[1:-2]:
			Efxk += pdf_t(i,df)
		
	outers = (pdf(x[0], df) 
					+ pdf(x[-1], df))/2
					
	res = round(abs(dx * (Efxk + outers))* tail, 5)
	
	if res <= p:
		print( f"Null hypothysis rejected! \nas p={res} and at p<{p}%, \nand {tail}-tailed probability distribution.")

	else:
		print(f"Null hypothysis accepted. \nas p={res} and at p<{p}%, \nand {tail}-tailed probability distribution.")
	
	return res


def t_indep(lst1, lst2):
	""" t value for the t test for 2 samples of independent means"""
	
	u1 = mean(lst1)
	u2 = mean(lst2)
	
	n1 = len(lst1)
	n2 = len(lst2)
	
	v1 = var(lst1)
	v2 = var(lst2)
	
	U= u1 - u2
	N = (n1+n2)/(n1*n2)
	V = (((n1-1)*v1) + ((n2-1)*v2))/(n1+n2-2)
	
	return U/(sqrt(V*N))
	
def sed(val1, val2, n1=1, n2=1):
	""" Standard Error Difference
	SED (M1-M2) = SQRT(v1/n1 + v2/n2)
	= SEM of 2 means
	is how much difference it is reasonable to accept between 2 sample means if the null hypothesis is true
	"""
	if  isinstance(val1, list):
		if isinstance(val2, list):
			v1 = var(val1)
			v2 = var(val2)
			n1 = len(val1)
			n2 = len(val2)
	else:
		v1 = val1
		v2 = val2
	return sqrt(v1/n1 + v2/n2)
			

def cimd_2st(x1, x2, t):
	"""
	confidence interval of mean difference
	2 sample t test
	x1= sample 1 list
	t= t critcal point of certain dof and CI
	"""
	md = (mean(x1)-mean(x2))
	r = t * sed(x1,x2)
	
	if md-r < md+r:
		return md-r, md+r
	else:
		return md+r, md-r

	
###T test for Dependent Samples###

def t_dep(x1, x2):
	"""
	T value for t test of dependent valies
	"""
	ED = sum([j - i for i,j in zip(x1, x2)])
	ED2 = sum([(j - i)**2 for i,j in zip(x1, x2)])
	if len(x1)==len(x2):
		n=len(x1)
	else:
		print("Two samples are not equal in number!")
		
	return ED/(sqrt(((n*ED2)- (ED)**2)/(n-1)))



###ANOVA###

def f_calc2(*args):
	""" args: the lists of data
	retruns: multiple values including first and second dof to used by f_value() for calcularions
		"""
	k = len(args)
	N= 0
	EEX=0
	EEX_2_N = 0 # ((EEX)^2)/n
	EE_X2 = 0
	E_EX_2_N = 0
	
	for i in args:
		n, ex, _, e_x2, ex_2_n = f_calc(i)
		N += n
		EEX += ex
		EE_X2 += e_x2
		E_EX_2_N += ex_2_n
		
	EEX_2_N = (EEX**2)/N
	df1= k - 1
	df2= N - k
	first = E_EX_2_N - EEX_2_N
	second = EE_X2 - E_EX_2_N
	third = EE_X2 - EEX_2_N
	
	return EEX_2_N, df1, df2, first, second, third
	
def f_calc(x):
		n= len(x)
		EX = sum(x)
		mu = mean(x)
		E_X2 = sum([i**2 for i in x]) # sum of each score squared E(X^2)
		EX_2_n = (EX**2)/n
		return n, EX, mu, E_X2, EX_2_n
		
def f_value(*args):
	_, b, c, d, g, _ = f_calc2(*args)
	first = d
	second = g
	df1= b
	df2=c
	return round((first/df1)/(second/df2), 3)

def eta2(*args):
	"""
	Effect size for ANOVA
	"""
	_, _, _, d, _, f = f_calc2(*args)
	return d/f


def beta(x,y):
	"""Beta funcrion"""
	return (gamma(x)*gamma(y))/gamma(x+y)
	
def f_pdf(x, d1, d2):
	a,b = d1/2, d2/2
	g1= ((d1*x)**d1)*(d2**d2)
	g2= (d1*x + d2)**(d1+d2)
	rv = beta(a,b)
	g3= x* rv
	return (sqrt(g1/g2))/g3

def f_cdf(a, b, d1, d2, n=1000):
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
			Efxk += f_pdf(x, d1, d2)
		
	outers = (f_pdf(ab[0], d1, d2) 
					+ f_pdf(ab[-1], d1, d2))/2
	
	return dx * ( Efxk + outers)



def f_table(**vars):
    
    """
    F-distribution table
    Parameters
    ----------
    df1 : float, degrees of freedom df1.
    d2 : float, degrees of freedom df2.
    p : TYPE, optional, probability. The default is 0.05.
    a : TYPE, optional,, the end point of the skewed tail. The default is 1000.
    n : TYPE, optional
        fractions of the area under the curve(dx). The default is 100000.

    Returns
    -------
    res : float
        F critical value.

    """
    
    d1 = vars.get('df1')
    d2 = vars.get('df2')
    p = vars.get('p', 0.05)
    a = vars.get('a', 1000)
    n = vars.get('n', 10**6)
    dx = a/n
    x = list(np.linspace(a, dx, n))
    y=0 #sum of the area
    c=0    # counter of dx
    for i in x:
        y+= dx * f_pdf(i, d1, d2)
        if y >= p:
            break
        c+=1
    res = a - (c * dx)
    return round(res,4)
    
    
def f_cur_1(d1, d2, p=1.0, a=1000, n = 10**6):
    """ calculate the extent of the the right end at with area under the curve = 1.0"""
    dx = a/n
    x = list(np.linspace(dx, a, n)) # 10**-2
    y=0 #sum of the area
    c=0    # counter of dx
    for i in x:
        y+= dx * f_pdf(i, d1, d2)
        z = round(y, 4) # (y,5)
        
        if z >= p:
            print("end")
            break
        c+=1
    res = (c * dx)
    return res


def reject_null_h_f(**vars):
	"""
	Null hypothesis rejected or not
	for ANOVA F TEST STATISTICS
	f_v:  obtained f_value,
	d1 = dof 1
	d2 = dof 2
	"""
	f_v = vars.get('f_v')
	d1 = vars.get('d1')
	d2 = vars.get('d2')
	p= vars.get('p', 0.05)
#	a = vars.get('a', 1000)
#	n = vars.get('n', 10**6)
	
	ft= f_table(**vars)
		
	if f_v > ft:
		return f"Null hypothysis rejected! \nas Obtrained f value = {f_v}, and critical f = {ft} at p < {p}%, and DOFs = {d1, d2}."
	else:
		return f"Null hypothysis accepted. \nas Obtrained f value = {f_v}, and critical f ={ft} at p < {p}%, and DOFs = {d1, d2}."

### CORRELATION ###

def dof_c(var):
	""" DOF of correlation = n - 2"""
	if isinstance(var, list):
		r= len(var)
	else:
		r = var
	
	return r -2
	
def r_crit(**vars):
	""" values of Correlation coefficient needed for rejection of null hypothesis (critical value of r)
	degrees of freedom = n of the pairs - 2
	p = risk of significance, default 0.05
	tail = tails of the distribution, default (2)
	"""
	df = vars.get('df', 1)
#	p = vars.get('p', 0.05)
#	tail = vars.get('tail', 2)
	#print(df, p, tail)
	t= t_table(**vars)
	return sqrt((t**2)/(t**2 + df))



def reject_null_h_all(val, func, **vars):
	"""
	Universal Null hypothesis rejected or not
	- val: the obtained value of significance representinh the research hypothesis
	funtion returning the critical value of rejection for a given dof(s), p value (risk of significance, default = 0.05), and no. of tails (tail default=2) as **kwargs arguments
	"""
	#print(vars)
	crit = func(**vars)
		
	if abs(val) > abs(crit):
		return f"Null hypothysis rejected! \nas Value obtained = {round(val,4)}, \nand Critical value= +/- {round(crit,4)}."
	else:
		return f"Null hypothysis accepted. \nas Value obtained = {round(val,4)}, \nand Critical value = +/- {round(crit,4)}."

def sums_r(x,y):
	""" Calcs for Correlation coefficient and Regression"""
	x2 = [i**2 for i in x]
	y2 = [i**2 for i in y]
	xy = [x*y for x,y in zip(x,y)]
	ex = sum(x)
	ey = sum(y)
	exy = sum(xy)
	ex2 = sum(x2)
	ey2 = sum(y2)
	n = len(x)
	return n, ex, ey, exy, ex2, ey2
	
	
def r_xy(x, y):
	"""Pearson Correlation coefficient"""
	n, ex, ey, exy, ex2, ey2 = sums_r(x,y)
	return (n*exy - ex*ey)/(sqrt((n*ex2 - ex**2)*(n*ey2 - ey**2)))


def reg(x,y):
	"""Linear regression:
		to predict a and b, in
		Y' = bX + a
	"""
	n, ex, ey, exy, ex2, _ = sums_r(x,y)
	b = ((exy - (ex*ey)/n)/(ex2 - ((ex**2)/n)))
	a = (ey - b*ex)/n
	return b,a


def r_2(r):
	"""Pearson's Coeff of determination
	indicating the amount of variance explained by the correlation
	r: pearson corr coeff
	"""
	return r**2

def p_r(d, rxy):
	""" 2 tail significance (probabilty) of correllation:
		d: dof of correlation n-2
		rxy: pearson correl coeff
		**probability of rejecting a null hypothesis while it is true = probaility of type 1 error
		"""
	if rxy < 0:
		i = -1
	t= sqrt((d)/((rxy**-2)-1))*i
	return t, p_of_t(t,d)
	
###CHI SQUARED###
def chi_pdf(df, x):
	a= (2**(df/2)) * (gamma(df/2))
	return (1/a) * (x **((df/2) -1)) * (e**(-x/2))

def chi_cdf(a, b, df, n=1000):
	dx= (b-a)/n
	ab= np.linspace(a,b,n)
	Efxk=0
	for x in ab[1:-2]:
			Efxk += chi_pdf(df, x)
		
	outers = (chi_pdf(df, ab[0]) 
					+ chi_pdf(df, ab[-1]))/2
	
	return dx * ( Efxk + outers)

def chi_range(df):
	""" 
	to return the upper an lower valies of significance of any chi squared curve according to dof
	"""
	#define the center of the chi curve
	# as a starting point
	if df <3:
		i = j = 0.1
	else:
		i = j = df-2
	g=7
	x =50
	while x > 10**-g:
		j+= 0.0001
		x = chi_pdf(df, j)
	x=50
	while x > 10**-g:
		i-= 0.0001
		if i < 0.00001: # prevent i from going to zero
			break
		x = chi_pdf(df, i)
	return i, j
	
def chi_crit(**vars):
	"""
	Chi square critical value table.
	 p:  probaility relayed to level of risk or level of significance (e.g. 0.05, 0.025)
	 df: degrees of freedom
	tail: 1= 1 tail, 2= 2 tailed
	 	
	 """
	df = vars.get('df')
	p = vars.get('p', 0.05)
	tail = vars.get('tail', 2)
	n= vars.get('n', 100000)
	
	p = p/tail
	i,j = chi_range(df)
	print(i,j)
	l = list(np.linspace(i, j, n)) 
	u = list(np.linspace(j, i, n)) 
	dx = (j-i)/n
	y=0
	a=b=0
	for il in l: # lower tail of curve
		y+= dx * chi_pdf(df, il)
		if y >= p:
			break
		a+=1
	y=0
	for iu in u: # upper tail
		y+= dx * chi_pdf(df, iu)
		if y >= p:
			break
		b+=1
		
	a = i + (a*dx)
	b = j - (b*dx)
	return round(a, 4), round(b, 4)
