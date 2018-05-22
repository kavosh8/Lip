import run_pacman, numpy
import planner
import sys
import random
import tensorflow as tf
import learn_tabular_models.learn_tabular_model as lt

for run_ID in range(2,10):
	model_params={}
	#model_params['lipschitz_constant']=float(sys.argv[1])
	model_params['run_ID']=int(run_ID)
	#model_params['learning_rate']=float(sys.argv[3])
	#model_params['gaussian_variance']=float(sys.argv[4])

	#make sure results are reproducable ...
	numpy.random.seed(run_ID)
	random.seed(run_ID)
	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
	from keras import backend as K
	tf.set_random_seed(run_ID)
	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	K.set_session(sess)
	#make sure results are reproducable ...
	lt.run(run_ID)
	pl=planner.planner(planner_type='tabular',model_params=model_params)
	temp=[]
	for j in range(10):
		temp.append(run_pacman.run(pl,num_time_steps=1000,show=False))
		numpy.savetxt('returns/tabular/'+str(run_ID)+".txt",temp)
