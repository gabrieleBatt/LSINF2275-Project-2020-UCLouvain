#Script Input: layout, circle
#python playTest.py <layout> <circle>
#python playTest.py 0 0 1 0 2 0 1 0 2 0 0 0 3 4 True
#Script Function: computes the optimal cost and policy
#with markov and simulaates the cost with the
#optimal policy simulating games
######################################################
import sys
import numpy as np
######################################################
from utils import *
from playSimulation import playSimulation
from markovDecision import *
######################################################

SQUARE_N = 15-1
NORM = 0

SAFETY = 1
RISKY = 2

N_SUMULATIONS = 10000

#simulates N_SIMULATIONS times with each of this as a starting cell
STARTING_CELL_LIST = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

######################################################
	
#init
options = sys.argv[1:]
if(len(options) == SQUARE_N + 1):
	layout = np.array(list(map(int, options[0:SQUARE_N])))
	circle = (options[SQUARE_N] == 'True')
else:
	layout = np.array([NORM] * (SQUARE_N))
	circle = True
	
dump("Input - layout", layout)
dump("Input - circle", circle)

#compute optimal policy and cost
expec, dice = markovDecision(layout, circle)
dump("Optimal policy", dice)
dump("Final costs", expec)

#simulate with optimal policy and comute average cost
sum = np.array([0.0]*len(STARTING_CELL_LIST))
for i in range(N_SUMULATIONS):
	sum += playSimulation(layout, circle, dice, starting_cell_list=STARTING_CELL_LIST)
avg_cost = sum/N_SUMULATIONS

#compute error
error = expec[STARTING_CELL_LIST-1] - avg_cost

#print results
dump("Average cost", avg_cost)
dump("Error", [float("%.3f"%e) for e in error])
dump("Max error", float("%.6f"%max(abs(error))))
	
