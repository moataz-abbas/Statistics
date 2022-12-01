#pylint:disable=W0622
from math import sqrt, pi, e, gamma
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


# median from a list (the center most value, or average of the 2 midway values)
def median(lst):
	n = len(lst)  # total no. of elements
	mid = int(n/2)  # get the mid point
	x = sorted(lst)  # sort the lst

	def even(lst_e, n):
		""" function to deal with even series"""
		return (lst_e[n - 1] + lst_e[n])/2

	def odd(lst_o, n):
		""" function to deal with odd series"""
		return lst_o[n]

	if n % 2 == 0:  # if clause to check if series is even or odd
		return even(x, mid)
	else:
		return odd(x, mid)
    

# mean from a list (sum of values over there total number)
def mean(lst):
	n = len(lst)
	return sum(lst)/n

# weighted mean, calculated by multiplying the unique values with there respective frequencies, and getting their average
def weighted_mean(lst):
	w = Counter(lst)
	return sum([x*y for x, y in w.items()])/len(lst)



def mode(lst):
	'''
	The most common occurance in a list
	'''
	
	if len(lst)==0:  # To check if lst is empty
		return "The lst is empty!"
	w = Counter(lst)
	_, i = w.most_common(1)[0]  # assign the most common frequency to variable 'i'
	# Set comprehension to get a list of unique frequencies, to be used to check if the list has more than one frequency of occuring values.
	v = {v for _, v in w.items()}
	# get the list of most common elements that occured 'i' times in the data set
	m = [x for x, y in w.items() if y == i]
	
	if len(v) > 1:  # check that we have a mode, and display it according to its number
		return m
	else:
		return None

# Variance

def var(lst):
	"""sample variance (n-1)"""
	mu = mean(lst)
	return sum((x - mu)**2 for x in lst)/(len(lst)-1)


def var_p(lst):
    """ population variance """
    mu = mean(lst)
    return sum((x - mu)**2 for x in lst)/(len(lst))

# Standard diviation:
def std_p(lst):
    """ population stdev """
    return np.sqrt(var_p(lst))


def std(lst):
	"""sample stdev"""
	return np.sqrt(var(lst))


def std_error(lst):
	"""
	Standard of Error:
	using a data list
	The standard error of the mean, or simply standard error, indicates how different the population mean is likely to be from a sample mean. It tells you how much the sample mean would vary if you were to repeat a study using new samples from within a single population.
	"""
	return std(lst)/(sqrt(len(lst)))


def dof(val):
	"""
	Degrees of freedom
	no. of elements in a sample - 1
	either provide the sample in alist format or the number of elements as a single number
	"""
	if isinstance(val, list):
		r = len(val)
	else:
		r = val

	return r - 1


def z_score(x, mu, s):
	""" 
	Z score:
	    The Z-score, or standard score, is the number of standard deviations a given data point lies above or below the mean. The mean is the average of all values in a group, added together, and then divided by the total number of items in the group.
	    x: value
	    mu: mean
	    s: standard divation
	
	"""
	return (x - mu)/s


def x_of_z(z, mu, s):
	""" x value of a Z score 
	calculate the value that is represented by a specific Z score,, i.e. inverse of Z score calculation
	"""
	return (z*s) + mu


def pdf_n(x, mu=0, s=1):
	"""
	Probability distribution function
	if used without re-assigning mu(mean =0),
	 and s(stdev = 1), it returns the standard
	 normal distribution curve
	"""
	return (1/(s*(sqrt(2 * pi))))*(e ** (-0.5 * ((x-mu)/s)**2))


def cdf_n(a, b, mu=0, s=1, n=10000):
	"""
	Area under normal distribution curve:
		(Z - table calculator)
	# a and b are the start and end values you want to predict the probability of happening
	# mu and s are the mean and std of the normal distribution curve,
	#     the default values draw a standard model of mean zero and std of 1.
	# n reflects the density of the plot to increase the accuracy of the results
	"""
	dx = (b-a)/n
	ab = np.linspace(a, b, n)
	Efxk = 0
	for x in ab[1:-2]:
		Efxk += pdf_n(x, mu, s)

	outers = (pdf_n(ab[0], mu, s)
           + pdf_n(ab[-1], mu, s))/2

	return dx * (Efxk + outers)


def cdf_n_from_mu(x):
	"""
	CDF of Z score from mean,
	of standard distribution curve
	"""
	return cdf_n(0, x, 0, 1, 5000)


def skew(lst):
    """ Measure of skewiness"""
    return (3*(mean(lst)-median(lst)))/std(lst)


def kurt(lst):
    """ Measure of kurtosis """
    m = mean(lst)
    s = std_p(lst)
    ex = [(z_score(x, m, s))**4 for x in lst]
    return mean(ex)-3


def ci_n(val, mu=0, s=1, n=10000):
	"""
	 Confidence interval: 2 tailed
	 val:  confidence interval in decimals (e.g. 0.95, 0.99)
	 mu: mean ,, s: stdev of the curve
	 returns the range of the mean under normal distrubution curve

	 """
	p = -(1 - val)/2
	a = -3.6
	x = list(np.linspace(a, 0, n))
	dx = a/n
	y = 0
	c = 0
	for i in x:
		y += dx * pdf_n(i)
		if y <= p:
			break
		c += 1

	res = round(a - (c * dx), 3)

	return x_of_z(res, mu, s), x_of_z(-res, mu, s)


def ci_saved(val):
	if val == 0.95:
		return -1.96, 1.96
	elif val == 0.975:
		return -2.237, 2.237
	elif val == 0.99:
		return -2.567, 2.567
	else:
		return "Invalid confidence interval!"


def sem(val, n=1):
    """
	Standard Error of the Mean (SEM):
		measures how far the sample mean (average) of the data
        is likely to be from the true population mean.
        The SEM is always smaller than the SD.
        val = either provided as a list of the variables in the sample, or the STD of a sample, 
        in addition to its size (n)
  >> Interpretition:
        Alarge standard error indicates that the estimate is imprecise. •A small standard error indicates that the estimate is precise. 
        The standard error is reduced, i.e. we obtain a more precise estimate, if: 
        • the size of the sample is increased
        • the data are less variable.
	"""
    if isinstance(val, list):
        s = std(val)
        n = len(val)
    else:
        s = val
        
    return s/sqrt(n)


def osz_test(x_bar, u, s, n):
	"""
	One sample Z-test: (one sample T test)
		used to compare mean of population and sample
        and return z value to be used with z or t tables
	x_bar : mean of the sample
	n : size of the sample
	s : std of sample
	u : population mean (or given as "Test value")

	"""
	return (x_bar - u)/sem(s, n)


def p_of_z(z):
	"""
	probability from periphery of curve tails,
	according to z score, either from left or right of a standard normal distribution curve
	"""
	if z < 0:
		a = -3.7
		return round(cdf_n(a, z, 0, 1), 4)
	else:
		b = 3.7
		return round(cdf_n(z, b, 0, 1), 4)


def reject_null_h(c, z):
	"""
	Null hypothesis rejected or not
	c = CI probability
	z = z score
	"""
	a, b = ci_n(c)
	if z < 0:
		r = -z+a
	else:
		r = z-b
	p = (1-c)/2
	if r >= 0:
		print("Null hypothesis REJECTED, \nStatistically Significant difference!"
		+ f"\np<{round(p, 3)}")
		return r
	else:
		print(
		    f"Null hypothesis ACCEPTED, \nNo Statistically significant difference!\np>{round(p, 3)}")
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
	""" 
	Probability distribution function of T tables
	x = x axis values
	v = degrees of freedom
	"""
	return ((gamma((v+1)/2))/(sqrt(v*pi) * gamma(v/2))) * ((1 + (x**2/v))**(-1*(v + 1)/2))


def t_table(**vars):
	"""
	 T* - T table, critical t value
	 Calculate the critical T value
	 vars is dict of the following values:
	 df = degrees of freedom
	 p= probaility relayed to level of risk or level of significance (e.g. 0.05( default value), 0.025)
	tail: 1= 1 tail, 2= 2 tailed (default=2)
	n = increases the accuracy of the calculation (default=10000), no need to change unless you want to decrease time of calculations
	 """
	df = vars.get('df')
	p = vars.get('p', 0.05)
	tail = vars.get('tail', 2)
	n = vars.get('n', 10000)

	p = -(p)/tail
	if df == 1:
		a = -770
	elif df == 2:
		a = -80
	elif df <= 5:
		a = -25
	else:
		a = -6
	x = list(np.linspace(a, 0, n))
	dx = a/n
	y = 0
	c = 0
	for i in x:
		y += dx * pdf_t(i, df)
		if y <= p:
			break
		c += 1

	res = round(a - (c * dx), 4)
	return -res


def reject_null_h_t(t, df, tail=2, p=0.05, n=10000):
	"""
	 Probability of the null hypothesis being true,,
	 null hypothysis will be rejected if p<0.05

	 t: obtained t value
	 df: degrees of freedom
	 p: level of risk

	 """
	if df == 1:
		a = -770
	elif df == 2:
		a = -80
	elif df <= 5:
		a = -25
	else:
		a = -6

	if t < 0:
		r = a - t
	else:
		a = -a
		r = a - t

	x = list(np.linspace(t, a, n))
	dx = r/n
	Efxk = 0
	for i in x[1:-2]:
			Efxk += pdf_t(i, df)

	outers = (pdf_t(x[0], df)
           + pdf_t(x[-1], df))/2

	res = round(abs(dx * (Efxk + outers)) * tail, 5)

	if res <= p:
		print(
		    f"Null hypothysis rejected! \nas p={res} and at p<{p}%, \nand {tail}-tailed probability distribution.")

	else:
		print(
		    f"Null hypothysis accepted. \nas p={res} and at p<{p}%, \nand {tail}-tailed probability distribution.")

	return res


def t_indep(lst1, lst2):
	""" t value for the t test for 2 samples of independent means"""

	u1 = mean(lst1)
	u2 = mean(lst2)

	n1 = len(lst1)
	n2 = len(lst2)

	v1 = var(lst1)
	v2 = var(lst2)

	U = u1 - u2
	N = (n1+n2)/(n1*n2)
	V = (((n1-1)*v1) + ((n2-1)*v2))/(n1+n2-2)

	return U/(sqrt(V*N))

def dof_t_indep(val):
    """
    Degrees of freedom for independent T-test 
    dof = both samples sizes (N)-2
    """
    if isinstance(val, list):
        r = len(val)
    else:
        r = val
    return r - 2

def sed(val1, val2, n1=1, n2=1):
	""" Standard Error Difference
	SED (M1-M2) = SQRT(v1/n1 + v2/n2)
	= SEM of 2 means
	is how much difference it is reasonable to accept between 2 sample means if the null hypothesis is true
	"""
	if isinstance(val1, list):
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
	r = t * sed(x1, x2)

	if md-r < md+r:
		return md-r, md+r
	else:
		return md+r, md-r


###T test for Dependent Samples###

def t_dep(x1, x2):
	"""
	T value for t test of dependent valies
	"""
	ED = sum([j - i for i, j in zip(x1, x2)])
	ED2 = sum([(j - i)**2 for i, j in zip(x1, x2)])
	if len(x1) == len(x2):
		n = len(x1)
	else:
		print("Two samples are not equal in number!")

	return ED/(sqrt(((n*ED2) - (ED)**2)/(n-1)))


###ANOVA###

def f_calc2(*args):
	""" args: the lists of data
	retruns: multiple values including first and second dof to used 
    by f_value() for calcularions
		"""
	k = len(args)
	N = 0
	EEX = 0
	EEX_2_N = 0  # ((EEX)^2)/n
	EE_X2 = 0
	E_EX_2_N = 0

	for i in args:
		n, ex, _, e_x2, ex_2_n = f_calc(i)
		N += n
		EEX += ex
		EE_X2 += e_x2
		E_EX_2_N += ex_2_n

	EEX_2_N = (EEX**2)/N
	df1 = k - 1
	df2 = N - k
	first = E_EX_2_N - EEX_2_N
	second = EE_X2 - E_EX_2_N
	third = EE_X2 - EEX_2_N

	return EEX_2_N, df1, df2, first, second, third


def f_calc(x):
		n = len(x)
		EX = sum(x)
		mu = mean(x)
		E_X2 = sum([i**2 for i in x])  # sum of each score squared E(X^2)
		EX_2_n = (EX**2)/n
		return n, EX, mu, E_X2, EX_2_n


def f_value(*args):
	_, b, c, d, g, _ = f_calc2(*args)
	first = d
	second = g
	df1 = b
	df2 = c
	return round((first/df1)/(second/df2), 3)


def dof_anov(*args):
	_, b, c, _, _, _ = f_calc2(*args)
	df1 = b
	df2 = c
	return df1, df2
	
def eta2(*args):
	"""
	Effect size for ANOVA
	"""
	_, _, _, d, _, f = f_calc2(*args)
	return d/f


def beta(x, y):
	"""Beta function"""
	return (gamma(x)*gamma(y))/gamma(x+y)


def f_pdf(x, d1, d2):
	''' 
	F distribution function
	x= x axis takes positive values only, 0 not included if d1=1, otherwise from zero to positive infinity
	d1, d2 = dof 1 and 2
	'''
	a = (d1)/2
	b= (d2)/2
	g1 = ((d1*x)**d1)*(d2**d2)
	g2 = (d1*x + d2)**(d1+d2)
	rv = beta(a, b)
	g3 = x * rv
	return (np.sqrt(g1/g2))/g3


def f_cdf(a, b, d1, d2, n=1000):
	"""
	Area under f function pdf
	# a and b are the start and end values you want to predict the probability of happening
	#     d1, d2 represents dof1 and dof2
	# n reflects the density of the plot to increase the accuracy of the results
	"""
	dx = (b-a)/n
	ab = np.linspace(a, b, n)
	Efxk = 0
	for x in ab[1:-2]:
		Efxk += f_pdf(x, d1, d2)

	outers = (f_pdf(ab[0], d1, d2)
           + f_pdf(ab[-1], d1, d2))/2

	return dx * (Efxk + outers)


def f_table(**vars):
    """
    F-distribution table
    Parameters
    ----------
    df1 : degrees of freedom df1.
    df2 : degrees of freedom df2.
    p : probability. The default is 0.05.
    a : the end point of the skewed tail. The default is 1000.
    n : fractions of the area under the curve(dx). The default is 100000.
    Returns
    -------
    res : F critical value.
    """
    d1 = vars.get('df1')
    d2 = vars.get('df2')
    p = vars.get('p', 0.05)
    a = vars.get('a', 1000)
    n = vars.get('n', 10**6)
    dx = a/n
    x = list(np.linspace(a, dx, n))
    y = 0  # sum of the area
    c = 0    # counter of dx
    for i in x:
        y += dx * f_pdf(i, d1, d2)
        if y >= p:
            break
        c += 1
    res = a - (c * dx)
    return round(res, 4)

def f_cur_1(d1, d2, p=1.0, a=1000, n=10**6):
    """ calculate the extent of the the right end at with area under the curve = 1.0"""
    dx = a/n
    x = list(np.linspace(dx, a, n))  # 10**-2
    y = 0  # sum of the area
    c = 0    # counter of dx
    for i in x:
        y += dx * f_pdf(i, d1, d2)
        z = round(y, 4)  # (y,5)

        if z >= p:
            print("end")
            break
        c += 1
    res = (c * dx)
    return res


def reject_null_h_f(**vars):
	"""
	Null hypothesis rejected or not
	for ANOVA F TEST STATISTICS
	f_v:  obtained f_value,
	df1 = dof 1
	df2 = dof 2
	"""
	f_v = vars.get('f_v')
	d1 = vars.get('df1')
	d2 = vars.get('df2')
	p = vars.get('p', 0.05)
#	a = vars.get('a', 1000)
#	n = vars.get('n', 10**6)

	ft = f_table(**vars)

	if f_v > ft:
		return f"Null hypothysis rejected! \nas Obtrained f value = {f_v}, and critical f = {ft} at p < {p}%, and DOFs = {d1, d2}."
	else:
		return f"Null hypothysis accepted. \nas Obtrained f value = {f_v}, and critical f ={ft} at p < {p}%, and DOFs = {d1, d2}."

### CORRELATION And Covariance ###


def dof_cor(vars):
	""" DOF of correlation = n - 2"""
	if isinstance(vars, list):
		r = len(vars)
	else:
		r = var
	return r - 2


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
	t = t_table(**vars)
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


def sums_r(x, y):
	""" Calcs for Correlation coefficient and Regression"""
	x2 = [i**2 for i in x]
	y2 = [i**2 for i in y]
	xy = [x*y for x, y in zip(x, y)]
	ex = sum(x)
	ey = sum(y)
	exy = sum(xy)
	ex2 = sum(x2)
	ey2 = sum(y2)
	n = len(x)
	return n, ex, ey, exy, ex2, ey2


def r_xy(x, y):
	"""
	Pearson Correlation coefficient:
	    Correlation is the standardized form of covariance by dividing the covariance with SD of each variable under normal distribution assumption. Generally, we use ‘r’ as sample correlation coefficient and ‘ρ’ as population correlation coefficient. The Pearson correlation coefficient has following formula.
	    r= cov(x,y)/(std(x)*std(y))
However, here it is calculated by another formula.

The Pearson correlation coefficient is also the covariance of standardized form of X and Y variables. The correlation coefficient is unit-less, being independent of the scale of variables and the range is between −1 and +1. The interpretation of the Pearson correlation coefficient was provided by Cohen [1]. He proposed a small, medium, and large effect size of r as 0.1, 0.3, and 0.5, respectively, and explained that the medium effect size represents an effect likely to be visible to the naked eye of a careful observer. Also he subjectively set a small effect size to be noticeably smaller than medium but not so small as to be trivial and set a large effect size to be the same distance above the medium as small was below it
	
	"""
	n, ex, ey, exy, ex2, ey2 = sums_r(x, y)
	return (n*exy - ex*ey)/(sqrt((n*ex2 - ex**2)*(n*ey2 - ey**2)))


def reg(x, y):
	"""Linear regression:
		to predict a and b, in
		Y' = bX + a
	"""
	n, ex, ey, exy, ex2, _ = sums_r(x, y)
	b = ((exy - (ex*ey)/n)/(ex2 - ((ex**2)/n)))
	a = (ey - b*ex)/n
	return b, a


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
	t = sqrt((d)/((rxy**-2)-1))*i
	return t, reject_null_h_t(t, d)

# Covariance
def cov(x,y):
    '''
    Covariance is defined as the expected value of variations of two variables from their expected values. More simply, covariance measures how much variables change together. The mean of each variable is used as reference and relative positions of observations compared to mean is important. Covariance is simply defined as the mean of multiplication of corresponding X and Y deviations from their mean. x - x_bar, y - y_bar.
    
    x= list of variables (numpy array)
    y= another list of variables (np array)
    
    this finds the correlation if it is linear?
    '''
    x_bar=mean(x)
    y_bar=mean(y)
    
    return ((sum((x-x_bar)*(y-y_bar)))/(len(x)-1))
    


###CHI SQUARED###


def chi_pdf(x, df):
	a = (2**(df/2)) * (gamma(df/2))
	return (1/a) * (x ** ((df/2) - 1)) * (e**(-x/2))


def chi_cdf(a, b, df, n=1000):
	dx = (b-a)/n
	ab = np.linspace(a, b, n)
	Efxk = 0
	for x in ab[1:-2]:
		Efxk += chi_pdf(df, x)

	outers = (chi_pdf(df, ab[0])
           + chi_pdf(df, ab[-1]))/2

	return dx * (Efxk + outers)


def chi_range(df):
	if df < 3:
		i = j = 0.1
	else:
		i = j = df-2
	g = 7
	x = 50
	while x > 10**-g:
		j += 0.0001
		x = chi_pdf(df, j)
	x = 50
	while x > 10**-g:
		i -= 0.0001
		if i < 0.0001:
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
	tail = vars.get('tail', 1)
	n = vars.get('n', 100000)

	p = p/tail
	i, j = chi_range(df)

	l = list(np.linspace(i, j, n))
	u = list(np.linspace(j, i, n))
	dx = (j-i)/n
	y = 0
	a = b = 0
	for il in l:
		y += dx * chi_pdf(df, il)
		if y >= p:
			break
		a += 1
	y = 0
	for iu in u:
		y += dx * chi_pdf(df, iu)
		if y >= p:
			break
		b += 1
	a = i + (a*dx)
	b = j - (b*dx)
	return round(a, 4), round(b, 4)


def chi_gof(**vars):
    """
    chi goodness of fit obtained value (X^2):
    ======================================
    accepts arguments as a list of the observed results (ob)
    and a list of expected results to compair (ex).
    or only observed results and the expected result will calculated,
    as equally divided proportions among the observed values 
    
    """
    ob = vars.get('ob')   #observed values
    ex = vars.get('ex', float(sum(ob)/len(ob))) #expected values
    
    if isinstance(ex, float):
        #convert the calculated expected value into
        #a list to fit the sum below
        n = len(ob)
        ex = [ex]*n
    
    X2 = sum([((o - e)**2)/e for o,e in zip(ob,ex)])
    return X2


def chi_indep(vars):
	"""
	Test of independence Chi-Square
	(chi-square test of association)
	Null hypothesis is accepted, in the case that the data sets are independent. i.e. the the added layer(s) of parameters doesnt affect the data distribution, = data is distributed randomely,, if the obtained value is less extreme than chi crit value.
	"""
	m = len(vars)
	n = len(vars[0])
	rs = [0]*m
	cs = [0]*n
	i = 0
	for r in vars:
		j = 0
		for c in r:
			rs[i] += c
			cs[j] += c
			j += 1
		i += 1
	es = sum(rs)  # total sum
	x = [rs, cs]
	ex = np.zeros((m, n))
	i = j = 0
	for i in range(m):
		for j in range(n):
			ex[i][j] = (x[0][i] * x[1][j])/es

	chi = [((vars[i][j]-ex[i][j])**2)/ex[i][j]
	        for i in range(m) for j in range(n)]
	return sum(chi)


def chi_dof(vars):
	if isinstance(vars[0], list):
			m = len(vars)
			n = len(vars[0])
			for i in range(m):
				assert len(
				    vars[i]) == n, "Rows of data should contain equal number of values!"
			return (m-1)*(n-1)
	else:
			return len(vars)-1


def chi_p(x, df, n=1000):
	""" Asymptotic significance (level of significance)
	returns the minimum of 2 levels of significance measured from both ends of the curve, to exclude the one measured away from the tails
	"""
	i, j = chi_range(df)
	a = i
	b = x
	c = j
	ab = np.linspace(a,b,n)
	bc = np.linspace(b,c,n)

	def auc(a, b, ab):
		dx = (b-a)/n
		Efxk = 0
		for x in ab[1:-2]:
			Efxk += chi_pdf(df, x)
		outers = (chi_pdf(df, ab[0])
            + chi_pdf(df, ab[-1]))/2
		return dx * (Efxk + outers)

	p1 = auc(a, b,ab)
	p2 = auc(b,c,bc)

	return min(p1, p2)
	
"""
ODDS RATIO
"""
	
def odds_r(a,b,c,d):
	"""
	Odds ratio:
		An odds ratio (OR) is a measure of association between an exposure and an outcome. The OR represents the odds that an outcome will occur given a particular exposure, compared to the odds of the outcome occurring in the absence of that exposure. Odds ratios are most commonly used in case-control studies, however they can also be used in cross-sectional and cohort study designs as well (with some modifications and/or assumptions).
		
		a= exposed cases
		b= exposed non cases
		c= unexposed cases
		d = unexposed and non cases
		
		a and c have the issue
		b and d  are good
		however the test marked
		b has an issue --> False +ve, type I error
		c as good --> False -ve, type II error
		
		OR = or odds for having the issue over the odds of not having the issue 
		= (a/c)/(b/d)
		or the odds of having the issue given the presence of the exposure / odds of having the issue in abscence of this exposure
		= (a/b)/(c/d)
		
		OR = ad/bc
		
		
		OR=1 Exposure does not affect odds of 				outcome

		OR>1 Exposure associated with higher odds 		of outcome

		OR<1 Exposure associated with lower odds 		of outcome
		"""
	return (a*d)/(b*c)
	
def odds_ci(**vars):
		
		a = vars.get('a')
		b = vars.get('b')
		c = vars.get('c')
		d = vars.get('d')
		ci_r = vars.get('ci', 0.95)
	
		o_r = odds_r(a,b,c,d)	
		o_z = ci_n(ci_r) #critical value of the confidence interval, returns the 2 values, the negative and positive value
		
		ln_or = np.log(o_r)
		sq= sqrt((1/a)+(1/b)+(1/c)+(1/d))
		
		def odd_form(z, ln, s):
			return  e ** (ln + (z * s))
		#lower ci
		l = odd_form(o_z[0], ln_or, sq)
		u = odd_form(o_z[1], ln_or, sq)
		
		return l, u
		
### PLOTS###

def boxplot_marks(d):
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
    
####Confusion Matrix####


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
    

