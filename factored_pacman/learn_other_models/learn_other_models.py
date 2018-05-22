import numpy 
import other_models
import csv
import sys


model_params={}
model_params["observation_size"]=6
model_params["num_hidden_layers"]=2
model_params["hidden_layer_nodes"]=32
model_params["activation_fn"]='relu'
model_params["learning_rate"]=0.001
other_models_object=other_models.neural_other_model(model_params)

def load_data(N):
	li_states=[]
	li_next_states=[]
	li_actions=[]
	li_rewards=[]
	li_dones=[]
	for i in range(N):
		x0,y0,x1,y1,x2,y2=numpy.random.randint(0,7,6)
		s=[x0,y0,x1,y1,x2,y2]
		a=numpy.random.randint(0,4)
		if a==0:
			#y0+1
			s_p=[x0,min(y0+1,6),x1,y1,x2,y2]
		elif a==1:
			#y0-1
			s_p=[x0,max(0,y0-1),x1,y1,x2,y2]
		elif a==2:
			#x0+1
			s_p=[min(x0+1,6),y0,x1,y1,x2,y2]
		elif a==3:
			#x0-1
			s_p=[max(0,x0-1),y0,x1,y1,x2,y2]
		if (s_p[0]==s_p[2] and s_p[1]==s_p[3]) or (s_p[0]==s_p[4] and s_p[1]==s_p[5]):
			r=-1
			done=1
			#print(s_p)
		else:
			r=0
			done=0
		a_temp=4*[0]
		a_temp[a]=1
		li_states.append(s),li_next_states.append(s_p),li_actions.append(a_temp),li_rewards.append(r),li_dones.append(done)
	return li_states,li_next_states,li_actions,li_rewards,li_dones



num_epochs=200
current_states_data,next_states_data,actions_data,rewards_data,dones_data=load_data(5*49*2)
other_models_object.ghost_model.fit(numpy.array(current_states_data)[:,2:4],numpy.array(next_states_data)[:,2:4],epochs=num_epochs)