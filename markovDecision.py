import sys
import numpy as np
from utils import *

def markovDecision(layout,circle):
	NORM = 0
	REST = 1
	PNLT = 2
	PRIS = 3
	MYST = 4

	SQUARE_N = 15-1
	WRAP_SQUARES = [10-1,14-1]
	LANE_ENTRY = [11-1,12-1,13-1]
	
	GO_BACK_PENALTY = 3
	LANE_CORRECTION = 7

	MOVE_COST = 1.0
	PRISON_PENALTY_COST = 1.0

	SAFETY = 1
	RISKY = 2

	INIT_COST = 100.0
	ITERATIONS = 50


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
	for i in range(ITERATIONS):
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
				
	return (expec, dice)
	
