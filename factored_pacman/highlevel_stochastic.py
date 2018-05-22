import run_pacman, numpy
import planner
import sys
import random
import tensorflow as tf

model_params={}
model_params['lipschitz_constant']=float(sys.argv[1])
model_params['run_ID']=int(sys.argv[2])
model_params['learning_rate']=float(sys.argv[3])
model_params['gaussian_variance']=float(sys.argv[4])


#make sure results are reproducable ...
numpy.random.seed(model_params['run_ID'])
random.seed(model_params['run_ID'])
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(model_params['run_ID'])
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#make sure results are reproducable ...

pl=planner.planner(planner_type='stochastic',model_params=model_params)
temp=[]
for _ in range(100):
	temp.append(run_pacman.run(pl,num_time_steps=1000,show=False))
	numpy.savetxt('returns/'+str(model_params['lipschitz_constant'])+\
		"-"+str(model_params['run_ID'])+"-"+str(model_params['learning_rate'])+\
		"-"+str(model_params['gaussian_variance'])+".txt",temp)