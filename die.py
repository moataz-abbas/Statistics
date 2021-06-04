from random import randint

class Die:
	def __init__(self, min= 1, max=6):
		self.min = min
		self.max = max
		self.num_sides = max - min
	
	def roll(self):
		if(self.num_sides>0):
			return randint(self.min, self.max)
		else:
			return 0
