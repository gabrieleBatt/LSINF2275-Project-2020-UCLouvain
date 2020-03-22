#Script for producing the bar plot in the report
######################################################
import sys
import numpy as np
import random as random
import matplotlib.pyplot as plt
######################################################
from utils import *
from playSimulation import playSimulation
from markovDecision import *
######################################################

NORM = 0
REST = 1
PNLT = 2
PRIS = 3
MYST = 4

SAFETY = 1
RISKY = 2

SQUARE_N = 15 - 1


N_ITER = 10000


def getIncrements(circle, trap):
	layout = [NORM]*SQUARE_N
	policy = [RISKY]*SQUARE_N
	avg_list = list()
	#baseline
	sum = 0.0
	for _ in range(N_ITER):
		sum += playSimulation(layout, circle, policy)[0]
	avg = sum/(N_ITER)	
	
	dump("trap %d - avg 0" % trap, avg)
	avg_list.append(avg)
	avg_list.append(0)
	
	for i in range(1,SQUARE_N):
		layout = [NORM]*SQUARE_N
		layout[i] = trap
		sum = 0.0
		for _ in range(N_ITER):
			sum += playSimulation(layout, circle, policy)[0]
		avg = sum/(N_ITER)	
		
		dump("trap %d - layout %d" % (trap,i), layout)
		dump("trap %d - avg %d" % (trap,i), avg)
		
		avg_list.append(avg)
	return avg_list
	
pris_avg = getIncrements(False, 3)
pnlt_avg = getIncrements(False, 2)
rest_avg = getIncrements(False, 1)
dump("Prisons", pris_avg)
dump("Penalty", pnlt_avg)	
dump("Restart", rest_avg)	

plt.bar(np.array(range(len(pris_avg)))-0.2, pris_avg, width=0.2, color='g', label='prison')
plt.bar(np.array(range(len(pnlt_avg))), pnlt_avg, width=0.2, color='b', label='penalty')
plt.bar(np.array(range(len(rest_avg)))+0.2, rest_avg, width=0.2, color='r', label='restart')

plt.legend(loc="upper left")
plt.ylabel('Average number of steps')
plt.xlabel('Number of traps')
plt.xticks([0,2,3,4,5,6,7,8,9,10,11,12,13])
plt.show()
	
		