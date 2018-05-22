
import numpy
import matplotlib.pyplot as plt
from rlenv.grid import grid_env as grid_env
import planner

def run(pl,num_time_steps,show):
	Env = grid_env(show)
	#print('Initialized, starting to train')
	
	s = Env.reset()
	num_dead=0
	for t in range(num_time_steps):
		a=pl.choose_action(s)
		s1,r,dead = Env.step([a])
		if show:
			plt.pause(0.05)
			Env.plot()
		if dead:
			s = Env.reset() 
			num_dead=num_dead+1
		else:
			s=s1
	return num_dead

