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
		self.fitness = []
		
		self.bestPop = [] 	#To store all 'best populations', b in the paper
		self.bestPop_cost = []	#and their respective costs
		
		'''
		self.group = np.empty((0, 2, 2), float)
		self.group = [self.group for i in range(self.bots)]
		
		
		# Initialize group
		for index, val in enumerate(self.lines):
			self.group[index % self.bots] = np.append(self.group[index % self.bots], np.array([val]), axis = 0)
		print(self.group)
		'''
		
		#length of the lines
		self.length = []
		for l in lines:
			dist = np.linalg.norm(l[0] - l[1])
			self.length.append(dist)
			
	# Initialize population
	def initPop(self):
		pop_size = 4
		self.population = np.random.random_integers(low = 0, high = self.bots-1, size = (pop_size, self.numLines))
		print ('\nPossible Solutions:\n',self.population)
		#First initialize individuals then stack them into a list
		#for i in range(self.numViduals):
			#vidual = (randomly initialize for first iteration)
			#self.population.append(self.population, vidual)


	# Return list of groups from a given individual
	def groupGenes(self, pop_sol):
		groups = []
		for i in range(self.bots):
			line_index = 0
			val = []
			for bot_index in pos_sol:
				if i == bot_index:
					val.append(self.lines[line_index])
				line_index = line_index + 1
			groups.append(val)
		# An individual is self.popolation[i], i < self.numViduals
		return groups

	# Function D in paper: returns dist b/w oldpose and next line
	def nextDist(self, pose, line, index_bot, chpose):
		x0 = pose[0]
		y0 = pose[1]
		x1 = line[0][0]
		y1 = line[0][1]
		x2 = line[1][0]
		y2 = line[1][1]
		
		d1 = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
		d2 = math.sqrt((x2 - x0)**2 + (y2 - y0)**2)
		
		if chpose == 1:
			if d1 < d2:
				end = [x2, y2]
				start = [x1, y1]
				return start,end
			else:
				end = [x1, y1]
				start = [x2, y2]
				return start, end
		dist = min(d1, d2)
		return dist
	
	# Return list of start/end points of closest line from current pose
	# Make sure start/end are in corect order
	def nextLine(self, pose, group, bot_index, line = []):
		dist = []
		for pts in group:
			distbtw = self.nextDist(pose = pose[bot_index], line = pts, bot_index = bot_index, chpose = 0)
			dist.append(distbtw)	#distance btw pose and all lines
		
		if len(dist) == 0:
			if (len(line) == 0):
				p = list(pose[bot_index])
				line.append(p)
			idx = None
			return line
		
		elif len(dist) == 1:
			if (len(line) == 0):
				p = list(pose[bot_index])
				line.append(p)
			idx = 0
			start, end = self.nextDist(pose = pose[bot_index], line = group[idx], bot_index = bot_index, chpose =1)
			line.append(start)
			line.append(end)
			return line 
		
		else:
			if (len(line) == 0):
				p = list(pose[bot_index])
				line.append(p)
	
			idx = dist.index(min(dist))
			start, end = self.nextDist(pose = pose[bot_index], line = group[idx], bot_index = bot_index, chpose = 1)
			line.append(start)
			line.append(end)
			
			pose[bot_index] = end
			group.remove(group[idx])
			line1 = self.nextline(pose, group, bot_index, line)
			return line				
		
	# Heuristics func in paper: Return distance/cost of a single input group
	def groupDist(self, line):
		cost = 0
		for index in range(len(line) - 1):
			pt1 = line[index]
			pt2 = line[index+1]
			dist = math.sqrt( ((pt1[0] - pt2[0])**2)+((pt1[1] - pt2[1])**2) )
			cost = cost + dist
		return cost


	# Returns cost of individual, i.e. Max(G1,G2,...)
	def groupCost(self, groupsCost):
		maxCost = max(groupsCost)
		return maxCost

	# Evalauate func in paper: Assigns best individual of the population
	def evaluate(self):
		best = min(self.fitness)
		best_idx = self.fitness.index(best)
		
		self.bestPop.append(self.population[best_idx])
		self.bestPop_cost.append(best)
		#self.bestPop = 


	# Alter function in paper
	def mutation(self):
	def crossover(self):


	# Main function: similar to fig 6 in paper.
	def main(self):
		self.initPop()
		
		const_pose = list(self.pose)
		for pos_sol in self.population:
			pose = np.array(const_pose) #initial pose should not change for diff soln
			
			groups = self.groupGenes(pos_sol)
			groupscost = []
			bot_index = 0
			for group in groups:
				line = self.nextline(pose, group, bot_index, line = [])
				bot_index = bot_index + 1
				print('line',line)
				
				c = self.groupDist(line)
				groupscost.append(c)
				print ('Next Group')
			
			maxCost = self.groupCost(groupscost)
			self.fitness.append(maxCost)
	
			print ('\n Next Population\n')
		print('Costs for the population:\n',self.fitness)
		self.evaluate()
		
		
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

https://github.com/beeclust-mrsl/Kim-Lee-MPS/blob/master/code.py
