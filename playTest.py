import sys
import numpy as np
import matplotlib.pyplot as plt
######################################################
from utils import *
from playSimulation import playSimulation
from markovTest import markovTest
######################################################

SQUARE_N = 15-1
NORM = 0

SAFETY = 1
RISKY = 2

N_SUMULATIONS = 100000

#simulates N_SIMULATIONS times with each of this as a starting cell
STARTING_CELL_LIST = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

######################################################
			
def playTest(layout, circle, policy, verbose=False, starting_cell_list=[1]):
	if verbose:
		dump("Input - layout", layout)
		dump("Input - circle", circle)
		dump("Input - policy", policy)

	#decision
	expec = playSimulation(layout, circle, policy, starting_cell_list=starting_cell_list)
	
	#if verbose:
		#dump("Final costs", expec)
		
	return expec

######################################################
	
#init
options = sys.argv[1:]
if(len(options) == SQUARE_N + 1):
	layout = np.array(list(map(int, options[0:SQUARE_N])))
	circle = (options[SQUARE_N] == 'True')
else:
	layout = np.array([NORM] * (SQUARE_N))
	circle = True

#compute optimal policy and cost
expec, dice = markovTest(layout, circle, verbose=False)

#simulate with optimal policy and comute average cost
sum = np.array([0.0]*len(STARTING_CELL_LIST))
for i in range(N_SUMULATIONS):
	sum += playTest(layout, circle, dice, verbose=False, starting_cell_list=STARTING_CELL_LIST)
avg_cost = sum/N_SUMULATIONS

#compute error
error = expec[STARTING_CELL_LIST-1] - avg_cost

#print results
dump("Average cost", avg_cost)
dump("Error", [float("%.3f"%e) for e in error])
dump("Max error", float("%.6f"%max(abs(error))))
	
