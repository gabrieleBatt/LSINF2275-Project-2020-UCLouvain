import sys
import numpy as np
import matplotlib.pyplot as plt
######################################################
from utils import *
from markovDecision import markovDecision
######################################################
NORM = 0
SQUARE_N = 15-1

######################################################
			
def markovTest(layout, circle, verbose=False):
	if verbose:
		dump("Input - layout", layout)
		dump("Input - circle", circle)

	#decision
	expec, dice = markovDecision(layout, circle)
	
	if verbose:
		dump("Optimal policy", dice)
		dump("Final costs", expec)
		
	return (expec, dice)

######################################################
	
#init
options = sys.argv[1:]
if(len(options) == SQUARE_N + 1):
	layout = np.array(list(map(int, options[0:SQUARE_N])))
	circle = (options[SQUARE_N] == 'True')
else:
	layout = np.array([NORM] * (SQUARE_N))
	circle = True

markovTest(layout, circle, verbose=True)
	
