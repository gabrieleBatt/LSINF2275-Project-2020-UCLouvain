#Script Input: layout, circle
#python playTest.py <layout> <circle>
#python playTest.py 0 0 1 0 2 0 1 0 2 0 0 0 3 4 True
#Script Function: computes the optimal cost
#with markov and plots the cost of each cell
#at each iteration (one color per cell)
######################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import *

SQUARE_N = 15-1

def markovDecisionPlot(layout,circle):
	NORM = 0
	REST = 1
	PNLT = 2
	PRIS = 3
	MYST = 4
	
	ITERATIONS = 1000
	PRECISION = np.array([1e-2]*SQUARE_N)
	
	WRAP_SQUARES = [10-1,14-1]
	LANE_ENTRY = [11-1,12-1,13-1]

	GO_BACK_PENALTY = 3
	LANE_CORRECTION = 7

	MOVE_COST = 1.0
	PRISON_PENALTY_COST = 1.0

	SAFETY = 1
	RISKY = 2

	INIT_COST = 100.0
	s_dice = np.array([	
			[1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1/2, 1/4, 0, 0, 0, 0, 0, 0, 1/4, 0, 0, 0, 0],
			[0, 0, 0, 1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 1/2, 1/2, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 1/2, 0, 0, 0, 0, 1/2],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/2, 1/2, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/2, 1/2],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			])
			
	r_dice = np.array([	
			[1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1/3, 1/6, 1/6, 0, 0, 0, 0, 0, 1/6, 1/6, 0, 0, 0],
			[0, 0, 0, 1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 1/3, 1/3, 1/3, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 1/3, 1/3, 0, 0, 0, 0, 1/3],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 1/3, 0, 0, 0, 0, 2/3],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/3, 1/3, 1/3, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/3, 1/3, 1/3, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/3, 1/3, 1/3],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/3, 2/3],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			])


	if(circle):
		for ws in WRAP_SQUARES:
			r_dice[ws][0] = 1/3
			r_dice[ws][SQUARE_N] = 1/3
	
	#Handle traps
	is_prison = np.array([(trap == PRIS) for trap in layout])
	is_mystery = np.array([(trap == MYST) for trap in layout])
	
	for i in range(0, len(layout)):
		#move probability to column 0
		if(layout[i] == REST):
			for j in range(0, len(layout)):
				r_dice[j][0] += r_dice[j][i]
				r_dice[j][i] = 0
		#move probability to three column to the left
		elif(layout[i] == PNLT):
			for j in range(0, len(layout)):
				#lane correction
				if(i in LANE_ENTRY):
					r_dice[j][max(0,i-LANE_CORRECTION-GO_BACK_PENALTY)] += r_dice[j][i]
				else:
					r_dice[j][max(0,i-GO_BACK_PENALTY)] += r_dice[j][i]
				r_dice[j][i] = 0
		#divide probability among three columns
		elif(layout[i] == MYST):
			for j in range(0, len(layout)):
				r_dice[j][0] += r_dice[j][i]*(1/3)
				#lane correction
				if(i in LANE_ENTRY):
					r_dice[j][max(0,i-LANE_CORRECTION-GO_BACK_PENALTY)] += r_dice[j][i]*(1/3)
				else:
					r_dice[j][max(0,i-GO_BACK_PENALTY)] += r_dice[j][i]*(1/3)
				r_dice[j][i] = r_dice[j][i]*(1/3)
			
	#inizialization	
	expec = np.array([INIT_COST] * SQUARE_N)	
	dice = np.array([SAFETY] * SQUARE_N)
	
	#until convergence
	j = 0
	old_expec = np.array([0] * SQUARE_N)
	
	#values for plot
	value_list = list()
	
	while((abs(old_expec - expec) > PRECISION).any() and j < ITERATIONS):
		j += 1
		value_list.append(list(expec))
		old_expec = np.array(expec)
		#Update policy
		##for each square policy
		for i in range(0, len(dice)):
			#compare dice costs
			s_cost = sum(s_dice[i][:-1]*expec)
			r_cost = sum(r_dice[i][:-1]*expec)
			if (s_cost < r_cost):
				dice[i] = SAFETY
			else:
				dice[i] = RISKY
		#Update cost
		##for each square cost
		for i in range(0, len(expec)):
			#if policy uses safety
			if(dice[i] == SAFETY):
				expec[i] = MOVE_COST + sum(s_dice[i][:-1]*expec)
			#if policy uses risky
			elif(dice[i] == RISKY):
				expec[i] = MOVE_COST + sum(r_dice[i][:-1]*expec)
				#prison cost increment
				expec[i] += PRISON_PENALTY_COST*sum(r_dice[i][:-1]*is_prison)
				#mystery cost increment
				expec[i] += PRISON_PENALTY_COST*(1/3)*sum(r_dice[i][:-1]*is_mystery)
				
	return (np.array(value_list).T,dice)
	
#init
options = sys.argv[1:]
if(len(options) == SQUARE_N + 1):
	layout = np.array(list(map(int, options[0:SQUARE_N])))
	circle = (options[SQUARE_N] == 'True')
else:
	layout = np.array([NORM] * (SQUARE_N))
	circle = True

	
value_list, dice = markovDecisionPlot(layout,circle)
    
colors = [(1,0,0), (0.5,0.5,0), (0,1,0), (0,0.5,0.5), (0,0,1), (0.5,0.5,0.5), (0,0,0), (0.75,0.25,0), (0.25,0.75,0), (0,0.25,0.75), (0,0.75,0.25), (0.5,0,0.5), (0.25,0,0.75), (0.75,0,0.25)]
for i in range(len(value_list)):
	plt.plot(np.array(range(1,len(value_list[i])+1)), value_list[i], '--', color = colors[i])

plt.title('Layout: %s - Circle: %s' % (layout, circle))
plt.show()

