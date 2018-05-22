import numpy
import utils
import sys

def load_synthetic_data(N):
	each=N/49
	li_s,li_sprime=[],[]
	for x in range(7):
		for y in range(7):
			s=[x,y]
			for _ in range(each):
				while True:
					case=numpy.random.randint(0,4)
					if case==0:
						s_p=[x+1,y]
					elif case==1:
						s_p=[x,y+1]
					elif case==2:
						s_p=[x-1,y]
					elif case==3:
						s_p=[x,y-1]
					if numpy.min(s_p)>=0 and numpy.max(s_p)<=6:
						break
				li_s.append(s)
				li_sprime.append(s_p)
	return li_s,li_sprime

def run(run_ID):
	states,next_states=load_synthetic_data(5*49)
	ghost_tabular_array=numpy.zeros(49*49).reshape((49,49))
	for x,y in zip(states,next_states):
		numberS=utils.state_2_number(x)
		numberSp=utils.state_2_number(y)
		ghost_tabular_array[numberS,numberSp]=ghost_tabular_array[numberS,numberSp]+1
	for index in range(ghost_tabular_array.shape[0]):
		ghost_tabular_array[index,:]=ghost_tabular_array[index,:]/(numpy.sum(ghost_tabular_array[index,:]))
	numpy.savetxt("learn_tabular_models/tabular"+str(run_ID)+".h5",ghost_tabular_array)