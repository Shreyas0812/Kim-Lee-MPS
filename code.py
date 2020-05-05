import numpy as np
import math

class kimLee():
	def __init__(self, numBots, lines, pose):
		self.lines = lines
		self.numLines = np.shape(self.lines)[0]
		self.bots = numBots
		self.pose = pose

		#self.numViduals = 
		self.population = []
		self.group = np.empty((0, 2, 2), float)
		self.group = [self.group for i in range(self.bots)]

		# Initialize group
		for index, val in enumerate(self.lines):
			self.group[index % self.bots] = np.append(self.group[index % self.bots], np.array([val]), axis = 0)
		print(self.group)
		
		#length of the lines
		self.length = []
		for l in lines:
			x1 = l[0][0]
			y1 = l[0][1]
			x2 = l[1][0]
			y2 = l[1][1]
			
			#print (x1,y1,x2,y2)
			dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
			self.length.append(dist)
			
	# Initialize population
	def initPop(self):
		self.population = np.random.random_integers(low = 1, high = self.numLines, size = (self.bots, self.numLines))
		print (self.population)
		#First initialize individuals then stack them into a list
		#for i in range(self.numViduals):
			#vidual = (randomly initialize for first iteration)
			#self.population.append(self.population, vidual)


	# Return list of groups from a given individual
	def groupGenes(self):
		# An individual is self.popolation[i], i < self.numViduals
		return groups


	# Return list of start/end points of closest line from current pose
	# Make sure start/end are in corect order
	def nextLine(self, pose):
		return line


	# Function D in paper: returns dist b/w oldpose and next line
	def nextDist(self, pose, line):
		return dist


	# Heuristics func in paper: Return distance/cost of a single input group
	def groupDist(self, group):
		return cost


	# Returns cost of individual, i.e. Max(G1,G2,...)
	def groupCost(self, groupsCost):
		return maxCost


	# Evalauate func in paper: Assigns best individual of the population
	def evaluate(self):
		#self.bestPop = 


	# Alter function in paper
	def mutation(self):
	def crossover(self):


	# Main function: similar to fig 6 in paper.
	def main(self):
		self.initPop()


# ToDo Later: Vizualizer function to plot/animate algorithm
class vizualizer():
	def plot(self):
	def animate(self):
	def vizualize(self):


if __name__ == '__main__':
	# Array of lines to be printed. Format: [(start_x, start_y), (end_x, end_y)]
	lineSet = np.array([
		[(1,11), (11,1)],
		[(2,22), (22,2)],               
		[(3,33), (33,3)],
		[(4,44), (44,4)],
		[(5,55), (55,5)]
		])

	# Initial position of each robot
	pose = np.array([(0,1), (1,2), (2,3)])
	test = kimLee(numBots = 3, lines = lineSet, pose = pose)
	test.main()


