from die import Die
from collections import Counter
import matplotlib.pyplot as plt
import stats_allstars as st
import numpy as np
import seaborn as sns

die = Die(1)

n=10000
y=[]
for i in range(n):
	x=[]
	for j in range(10):
		x.append(die.roll())
	y.append(sum(x))
	

#print(y)
#c= Counter(y)
#c=dict(sorted(c.items(), key= lambda i: i[0]))

#x = list(c.keys())
#y = list(c.values())
#print(x,y)
#fig, ax = plt.subplots()
#ax.bar(c.keys(), c.values())
#ax.plot(x, y, color='r', markersize=14)
#ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%d' % int(height), ha='center', va='bottom')

#rects = ax.patches
#labels = [i/n for i in c.values()]

#for rect, label in zip(rects, labels):
#	height = rect.get_height()
#	ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')

#n, bins, patches =  plt.hist(y, bins=11, density= True)
#print(n, bins)

plt.plot(bins,[st.pdf(i, st.mean(bins), st.std(bins)) for i in range(len(bins)) ])
mu = st.mean(y)
std = st.std(y)
x= np.linspace(0,10, len(y))
plt.plot(x, [st.pdf(i, mu, std) for i in x], color='r', linewidth=2.5)
print(st.pdf(mu,mu,std))
#plt.plot(bins, [st.cdf(0,i,mu,std) for i in bins], color='b', linewidth=2.5)

sns.histplot(y, bins=11, stat='density', discrete=True, kde=True,kde_kws={'bw_adjust' : 3})
plt.show()
