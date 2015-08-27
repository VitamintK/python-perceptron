import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
from collections import namedtuple

class Perceptron:
	def __init__(self, sample = None, dimensions = 2):
		self.d = dimensions
		self.w = np.zeros(dimensions)
		self.sample = sample
		self.b = 0
		self.done = False
		self.iterations = 0
	def iterate(self, plot = False):
		self.iterations += 1
		random.shuffle(self.sample)
		for x in self.sample:
			if (np.dot(x.x, self.w) > -self.b) != x.y:
				self.w = self.w + ((x.y)*2-1) * x.x
				self.b += (x.y)*2 - 1
				print(self.iterations, self.w, self.b)
				break
		else:
			print('we done here fAM')
			self.done = True
			return False
		if plot:
			self.plot()
			return True
	def learn(self, plot=False):
		while self.iterate(plot):
			pass
	def plot(self):
		assert(self.d == 2)
		plt.clf()
		plt.axis([-1,1,-1,1])
		for x in self.sample:
			#if (np.dot(x.x, self.w) >= -self.b) != x.y:
			if x.y != 1:
				plt.plot(x.x[0], x.x[1], 'or')
			else:
				plt.plot(x.x[0], x.x[1], 'ob')
		n = np.linalg.norm(self.w)
		if self.w[1] != 0:
			ww = 1.25 * self.w/n
			m = -self.w[0]/self.w[1]
			b = -self.b/self.w[1]
			l = np.linspace(-1,1)
			plt.plot(l, m*l + b, '-k')
			#l = np.linspace(-1,1)
			plt.fill_between(l, m*l+b, math.copysign(1, self.w[1]), alpha=0.5)
			plt.fill_between(l, m*l+b, math.copysign(1, -1*self.w[1]), color='r', alpha=0.5)

Point = namedtuple('Point', ['y', 'x'])

class Separable_Data:
	def __init__(self, n = 1000, f = None, dimensions = 2, bounds = None):
		if bounds == None:
			bounds = [(-1,1) for _ in range(dimensions)]
		if f == None:
			f = self.make_f()
		self.d = dimensions
		self.bounds = bounds
		self.f = f
		self.data = []
		self.generate_points(n)
	def make_f(self):
		up = random.randint(0,1)
		x = random.random()*2 - 1
		y = random.random()*2 - 1
		m = math.tan(random.random() * 2 * 3.14)
		print("m = {}, b = {}".format(m, -m*x + y))
		if up:
			return lambda x0, x1:  x1 - m*x0 < -m*x + y
		else:
			return lambda x0, x1:  x1 - m*x0 > -m*x + y
	def generate_points(self, n):
		for i in range(n):
			point = np.array([random.uniform(self.bounds[j][0], self.bounds[j][1]) for j in range(self.d)])
			self.data.append(Point(self.f(*point), point))
	def produce_sample(self, n):
		return random.sample(self.data, n)


plt.axis([-1,1,-1,1])
data = Separable_Data().produce_sample(70)
p = Perceptron(sample = data)

fig = plt.figure()
def animate(i):
	p.iterate(plot=True)
def g(repeats = 3):
	while not p.done:
		yield True
	for _ in range(0,repeats):
		yield True
ani = animation.FuncAnimation(fig, animate, frames = g(6), interval = 1000, save_count = 300)
#saving as GIF requires imagemagick to be installed and configured (on windows, by changing the rcsetup.py file http://stackoverflow.com/a/31869370)
#ani.save('PERCEPTRON-{}.gif'.format(int(time.time())), writer="imagemagick", fps=2)
#saving as mp4 requires ffmpeg
ani.save('PERCEPTRON-{}.mp4'.format(int(time.time())), fps=2)
print("finished saving")
plt.show()