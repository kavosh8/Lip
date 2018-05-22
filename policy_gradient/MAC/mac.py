import numpy
import value_network
import actor_network
import gym
import sys
import utils
import tensorflow as tf
import csv
import t
import math
gamma=0.99999

def interactOneEpisode(env,actor_q_network,sess,environment_name,max_time_steps,test):
	rep,rewards,reps,reps_prime,actions,t,probs=env.reset(),[],[],[],[],0,[]
	reps=[]
	while True:
		action,prob=actor_q_network.action_selection(rep,sess)
		rep_prime,r,done,_= env.step(action)
		#sys.exit(1)
		reps.append(rep),reps_prime.append(rep_prime),actions.append(action),rewards.append(r),probs.append(prob)
		rep,t=(rep_prime,t+1)
		if done==True or t>=max_time_steps:
			break
	returns=utils.rewardToReturn(rewards,gamma)
	if test:
		return returns[0]
	return returns,reps,reps_prime,actions,rewards,probs

def train(run,episode_batch_size,max_learning_episodes,max_time_steps,
			q_num_hidden_layers,q_hidden_layer_size,q_learning_rate,q_nb_epoch,
			p_num_hidden_layers,p_hidden_layer_size,p_learning_rate,environment_name,environment_name_test,q_batch_size,shuffle):
	env_test=gym.make(environment_name_test)
	#env_test=gym.make('LunarLander-v2')
	env=gym.make(environment_name)
	with tf.Session() as sess:
		
		state_size=env.observation_space.shape[0]#number of state variables
		action_size=env.action_space.n#number of actions
		q_network=value_network.network(state_size,action_size,q_num_hidden_layers,
									q_hidden_layer_size,q_learning_rate,q_nb_epoch,q_batch_size,shuffle,sess)#define Q network
		actor=actor_network.network(state_size,action_size,q_network,
												p_num_hidden_layers,p_hidden_layer_size,p_learning_rate,sess)# define policy + Q network
		saver= tf.train.Saver(actor.weights)
		init = tf.global_variables_initializer()
		sess.run(init)
		reps_batch,reps_all=[],[]
		reps_prime_batch,reps_prime_all=[],[]
		returns_list=[]
		return_per_episode=[]
		Gs_list=[]
		actions_batch,actions_all=[],[]
		for episode in range(1,max_learning_episodes+1):
			returns,reps,reps_prime,actions,_,probs=interactOneEpisode(env,actor,sess,environment_name,max_time_steps,False)#interact with the environment for one episode
			reps_batch=reps_batch+reps
			reps_prime_batch=reps_prime_batch+reps_prime
			#print(reps_batch[1])
			#print(reps_prime_batch[0])
			actions_batch=actions_batch+utils.actions21hot(actions,action_size)
			
			returns_list=returns_list+returns
			
			
			return_per_episode.append(returns[0])
			if episode%episode_batch_size==0:# do batch update, first the value network then the policy network
				q_network.update(returns_list,reps_batch,actions_batch)#update q this function is shuffling lists :|
				actor.update(reps_batch)#and then actor
				reps_all=reps_all+reps_batch
				reps_prime_all=reps_prime_all+reps_prime_batch
				actions_all=actions_all+actions_batch
				reps_batch,returns_list,actions_batch,reps_prime_batch=[],[],[],[]
			print_batch_size=100
			if episode%print_batch_size==0:
				print("episode #:",episode,"return average in trained environment",environment_name,numpy.mean(return_per_episode[-print_batch_size:]))
				Gs=[interactOneEpisode(env_test,actor,sess,'CartPole-v0',max_time_steps,True) for t in range(10)]
				print("return average in test environment",numpy.mean(Gs))
				Gs_list.append(numpy.mean(Gs))
				#sys.exit(1)
				sys.stdout.flush()
		print("end of training")
		numpy.savetxt("log/returns_train_"+environment_name+"_test_"+'CartPole-v0'+"_"+str(run)+"_on_train"+".txt",return_per_episode)
		numpy.savetxt("log/returns_train_"+environment_name+"_test_"+'CartPole-v0'+"_"+str(run)+"_on_test"+".txt",Gs_list)
		#saver.save(sess,'./log/policy_model_'+str(run))
	'''
	with open("log/reps_"+str(run)+".csv", "wb") as f:
		writer = csv.writer(f)
		print(len(reps_all))
		writer.writerows(reps_all)

	with open("log/reps_prime_"+str(run)+".csv", "wb") as f:
		writer = csv.writer(f)
		print(len(reps_prime_all))
		writer.writerows(reps_prime_all)
	with open("log/actions_"+str(run)+".csv", "wb") as f:
		writer = csv.writer(f)
		print(len(actions_all))
		writer.writerows(actions_all)
	'''