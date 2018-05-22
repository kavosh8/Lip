import mac
 # no need to tune this
import sys

try:
	run=sys.argv[1]
	environment=sys.argv[2]
	environment_name_test=sys.argv[3]
except:
	run=0

#best hyper parameters
episode_batch_size=10
q_num_hidden_layers=1
q_hidden_layer_size=50
q_learning_rate=0.05
q_nb_epoch=10
p_num_hidden_layers=1
p_hidden_layer_size=75
p_learning_rate=0.005
#environment='CartPole-v0'
q_batch_size=64
shuffle='every'
#best hyper parameters

max_learning_episodes=1000
max_time_steps=200


mac.train(run,episode_batch_size,max_learning_episodes,max_time_steps,
			q_num_hidden_layers,q_hidden_layer_size,q_learning_rate,q_nb_epoch,
			p_num_hidden_layers,p_hidden_layer_size,p_learning_rate,
			environment,environment_name_test,q_batch_size,shuffle)
