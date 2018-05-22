import numpy, sys
import learn_ghost_model.transition_model
import learn_other_models.other_models
import learn_tabular_models.tabular_model


class planner:
	em_model_object=0
	other_models_object=0


	def __init__(self,planner_type,model_params):
		self.type=planner_type # this determines what kind of model to use
		temp=model_params['run_ID']
		#*** 
		# parameters necessary to load learned EM model for ghosts
		model_params['hidden_layer_nodes']=32
		model_params['activation_fn']='relu'
		model_params['observation_size']=2
		model_params['num_models']=4
		model_params['num_epochs']=5
		model_params['num_hidden_layers']=2
		model_params['num_samples']=49*5
		if planner_type=='stochastic':
			#load=True
			fname='learn_ghost_model/log/model-'+str(model_params['run_ID'])+"-"+str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
		  	"-"+str(model_params['gaussian_variance'])+"-"+str(model_params['num_hidden_layers'])+"-"+str(model_params['lipschitz_constant'])
		  	# load EM model for ghosts
			self.em_model_object=learn_ghost_model.transition_model.neural_transition_model(model_params,True,fname)
		


		#parameters for loading deterministic models, including for ghost, reward, and pacman
		model_params={}
		model_params["observation_size"]=6
		model_params["num_hidden_layers"]=2
		model_params["hidden_layer_nodes"]=32
		model_params["activation_fn"]='relu'
		model_params["learning_rate"]=0.005
		model_params['run_ID']=temp
		fname='learn_other_models/'
		#load deterministic models ... True means actually load ...
		self.other_models_object=learn_other_models.other_models.neural_other_model(model_params,True,fname)

		#load tabular model ... there is no parameter for it, it is just a matrix
		fname='learn_tabular_models/'
		self.tabular_model_object=learn_tabular_models.tabular_model.tabular_model(fname,True,model_params['run_ID'])

	def predict_ghost_next_states_stochastic(self,ghosts):
		'''
		this function takes location of ghosts as input
		and outputs two lists. First is the next location of ghosts and second is their probability.
		both lists have the size 4*4=16.
		'''
		ghost1=numpy.array(ghosts[:2]).reshape(1,2)
		ghost2=numpy.array(ghosts[2:4]).reshape(1,2)
		t1=self.em_model_object.predict(ghost1)
		t2=self.em_model_object.predict(ghost2)
		li_location=[]
		li_probs=[]
		probs=self.em_model_object.probs
		for index1,x in enumerate(t1):
			for index2,y in enumerate(t2):
				temp=x[0].tolist()+y[0].tolist()
				li_location.append(temp)
				li_probs.append(probs[index1]*probs[index2])
		return li_location,li_probs

	def predict_stochastic(self,state,action):
		'''
		returns Q(state,action) using the em model for ghosts, and deterministic reward and pacman models
		returns just a single number computed by Q=probs*rewards(nextStates)
		'''
		ghost_next_locs,ghost_next_probs=self.predict_ghost_next_states_stochastic(state[2:])
		action_array=numpy.array(4*[0]).reshape(1,4)
		action_array[0,action]=1
		pacman_state_array=numpy.array(state[:2]).reshape(1,2)
		pacman_next_loc=self.other_models_object.pacman_model.predict([pacman_state_array,action_array])

		next_states=[pacman_next_loc[0].tolist()+ gnl for gnl in ghost_next_locs]
		next_rewards=self.other_models_object.reward_model.predict(next_states)
		next_rewards=next_rewards.tolist()

		Q=numpy.mean([nr[0]*gnp for (nr,gnp) in zip(next_rewards,ghost_next_probs)])
		return Q

	def predict_deterministic(self,state,action):
		'''
		returns Q(state,action) using only deterministic models, including for ghosts!
		'''
		ghost1_next_loc=self.other_models_object.ghost_model.predict(numpy.array(state[2:4]).reshape(1,2))[0].tolist()
		ghost2_next_loc=self.other_models_object.ghost_model.predict(numpy.array(state[4:6]).reshape(1,2))[0].tolist()
		
		action_array=numpy.array(4*[0]).reshape(1,4)
		action_array[0,action]=1
		pacman_state_array=numpy.array(state[:2]).reshape(1,2)
		pacman_next_loc=self.other_models_object.pacman_model.predict([pacman_state_array,action_array])[0].tolist()
		next_state=pacman_next_loc+ghost1_next_loc+ghost2_next_loc
		next_reward=self.other_models_object.reward_model.predict(numpy.array(next_state).reshape(1,6))
		Q=next_reward[0,0]
		return Q

	def predict_tabular(self,state,action):
		'''
		returns Q(state,action) using tabular model for ghosts and deterministic models for reward and pacman!
		'''
		ghosts_next_states,ghosts_next_probs=self.tabular_model_object.predict(state)
		action_array=numpy.array(4*[0]).reshape(1,4)
		action_array[0,action]=1
		pacman_state_array=numpy.array(state[:2]).reshape(1,2)
		pacman_next_loc=self.other_models_object.pacman_model.predict([pacman_state_array,action_array])[0].tolist()
		next_states=[pacman_next_loc+gns for gns in ghosts_next_states]
		#print(next_states)
		#print(len(next_states))
		next_states=numpy.array(next_states).reshape(len(next_states),6)
		next_reward=self.other_models_object.reward_model.predict(next_states)
		Q=numpy.mean([nr[0]*gnp for (nr,gnp) in zip(next_reward,ghosts_next_probs)])
		return Q

	def action_values(self,state):
		'''
		computes a list of action values q_li for the input state.
		the way in which the list is computed depends upon self.type
		there are three types: tabular method, deterministic method, stochastic (em) method
		'''
		q_li=[]
		for action in range(4):
			if self.type=='stochastic':
				Q=self.predict_stochastic(state,action)
				q_li.append(Q)
			elif self.type=='deterministic':
				Q=self.predict_deterministic(state,action)
				q_li.append(Q)
			elif self.type=='tabular':
				Q=self.predict_tabular(state,action)
				q_li.append(Q)
		#print(q_li)
		return q_li
		
	def choose_action(self,s,epsilon=0):
		'''
			given a state s, compute Q values and return max
			if policy is random, just randomly choose sth
			also if values are equal, make sure we choose randomly
		'''
		if self.type=='random':
			return numpy.random.randint(4)
		Qs=self.action_values(s)
		if numpy.max(Qs)==numpy.min(Qs):
			return numpy.random.randint(4)
		if numpy.random.random()>epsilon:
			#print(Qs)
			return numpy.argmax(Qs)
		else:
			return numpy.random.randint(4)


