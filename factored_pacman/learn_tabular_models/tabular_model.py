import numpy
import utils
import sys

class tabular_model:
	model=[]
	def __init__(self,fname,load,run_ID):
		if load==True:
			self.load_model(fname,run_ID)


	def load_model(self,fname,run_ID):
		self.model=numpy.loadtxt(fname+'tabular'+str(run_ID)+'.h5')
	def predict(self,state):
		ghost1_state_number=utils.state_2_number(state[2:4])
		ghost2_state_number=utils.state_2_number(state[4:6])
		ghost1_next=[]
		ghost1_next_probs=[]
		ghost2_next=[]
		ghost2_next_probs=[]
		for index,x in enumerate(self.model[ghost1_state_number]):
			if x>0:
				ghost1_next.append(index)
				ghost1_next_probs.append(x)
		for index,x in enumerate(self.model[ghost2_state_number]):
			if x>0:
				ghost2_next.append(index)
				ghost2_next_probs.append(x)
		li=[]
		li_probs=[]
		for index1,g1 in enumerate(ghost1_next):
			for index2,g2 in enumerate(ghost2_next):
				g1_state=utils.number_2_state(g1)
				g2_state=utils.number_2_state(g2)
				li.append(g1_state+g2_state)
				li_probs.append(ghost1_next_probs[index1]*ghost2_next_probs[index2])
		#print(li,li_probs)
		#sys.exit(1)
		if li==[]:
			li.append([0,0,0,0])
			li_probs.append([1])
		#print(li)
		return li,li_probs
