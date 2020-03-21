import sys
import numpy as np
import random
from utils import *

#simulates one game and computes the cost of the specified cells
def playSimulation(layout, circle, policy, starting_cell_list=[1]):
	NORM = 0
	REST = 1
	PNLT = 2
	PRIS = 3
	MYST = 4

	MOVE_COST = 1.0
	PRISON_PENALTY_COST = 1.0

	SAFETY = 1
	RISKY = 2
	
	#Trap detection
	is_restart = np.array([(trap == REST) for trap in layout])
	is_penalty = np.array([(trap == PNLT) for trap in layout])
	is_prison = np.array([(trap == PRIS) for trap in layout])
	is_mystery = np.array([(trap == MYST) for trap in layout])
	
	
	#init
	expec = np.array([0]*len(starting_cell_list))
	
	#for each cell as starting cell
	for i in range(len(starting_cell_list)): 
		turns = 0
		current_cell = starting_cell_list[i]
		while(current_cell != 15):
			turns += 1
			#if rolling saefty dice
			if(policy[current_cell-1] == SAFETY):
				#if roll dice is 1
				if(random.randint(0, 1) == 1):
					#lane choiche
					if(current_cell == 3):
						lane = random.randint(0, 1)
						current_cell = 4*lane + 11*(1-lane)
					#10 -> 15
					elif(current_cell == 10):
						current_cell = 15
					#default +1
					else:
						current_cell = current_cell + 1
			elif(policy[current_cell-1] == RISKY):
				roll = random.randint(0, 2)
				if(roll == 1):
					#lane choiche
					if(current_cell == 3):
						lane = random.randint(0, 1)
						current_cell = 4*lane + 11*(1-lane)
					#10 -> 15
					elif(current_cell == 10):
						current_cell = 15
					#default +1
					else:
						current_cell += 1
				elif(roll == 2):
					#lane choiche
					if(current_cell == 3):
						lane = random.randint(0, 1)
						current_cell = 5*lane + 12*(1-lane)
					#9 -> 15
					elif(current_cell == 9):
						current_cell = 15
					#roll over
					elif(current_cell == 10 or current_cell == 14):
						if (circle == True):
							current_cell = 1
						else:
							current_cell = 15
					#default +1
					else:
						current_cell += 2
						
						
				#Traps
				if(current_cell != 15):
					#go back to start
					if(is_restart[current_cell-1]):
						current_cell = 1
					#go back three
					elif(is_penalty[current_cell-1]):
						if(current_cell == 11 or current_cell == 12 or current_cell == 13):
							current_cell = max(1, current_cell-7-3)
						else:
							current_cell = max(1, current_cell-3)
					#skip next turn
					elif(is_prison[current_cell-1]):
						turns += 1
					#mystery trap
					elif(is_mystery[current_cell-1]):
						trap = random.randint(1, 3)
						#go back to start
						if(trap == REST):
							current_cell = 1
						#go back three
						elif(trap == PNLT):
							if(current_cell == 11 or current_cell == 12 or current_cell == 13):
								current_cell = max(1, current_cell-7-3)
							else:
								current_cell = max(1, current_cell-3)
						#skip next turn
						elif(trap == PRIS):
							turns += 1
				
			
		expec[i] = turns
				
	return expec
	
