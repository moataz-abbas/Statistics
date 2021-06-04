from die import Die
from collections import Counter
import matplotlib.pyplot as plt


die = Die(1)

n=10000
y=[]
for i in range(n):
	x=[]
	for j in range(10):
		x.append(die.roll())
	y.append(sum(x))
	

print(y)
c= Counter(y)
c=dict(sorted(c.items(), key= lambda i: i[0]))
#print(c.values())


fig, ax = plt.subplots()
ax.bar(c.keys(), c.values())

#ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%d' % int(height), ha='center', va='bottom')

rects = ax.patches
labels = [i/n for i in c.values()]

for rect, label in zip(rects, labels):
	height = rect.get_height()
	ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')


plt.show()
