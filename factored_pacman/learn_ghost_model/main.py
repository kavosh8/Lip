import numpy
import numpy.random
import sys, random
import tensorflow as tf
import numpy.linalg as lg
import matplotlib.pyplot as plt
import sys
import transition_model
import math
import em
import keras
import scipy.stats
import time
import csv
import utils



# take as input, from qsub script, run ID num samples stepsize and gaussian variance ...

model_params={}
em_params={}
try:
	
	model_params['lipschitz_constant']=float(sys.argv[6])
	model_params['num_hidden_layers']=int(sys.argv[5])
	em_params['gaussian_variance']=float(sys.argv[4])
	model_params['learning_rate']=float(sys.argv[3])
	model_params['num_samples']=int(sys.argv[2])
	run_number=int(sys.argv[1])
except:
	print("did not input necessary hyper-parameters")
	model_params['lipschitz_constant']=.01
	model_params['num_hidden_layers']=2
	em_params['gaussian_variance']=.1
	model_params['learning_rate']=0.005
	model_params['num_samples']=5*49
	run_number=0
	print("trying default ones instead")

model_params['hidden_layer_nodes']=32
model_params['activation_fn']='relu'
model_params['observation_size']=2
model_params['num_models']=4
model_params['num_epochs']=5
em_params['num_iterations']=500
em_params['num_models']=model_params['num_models']
em_params['observation_size']=model_params['observation_size']



#make sure results are reproducable ...
numpy.random.seed(run_number)
random.seed(run_number)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(run_number)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#make sure results are reproducable ...

li_w,li_em_obj=[],[]
#build training data
li_samples,li_labels=utils.load_synthetic_data(model_params['num_samples'])
phi,y=utils.create_matrices(li_samples,li_labels,model_params)
#create transition model
tm=transition_model.neural_transition_model(model_params)
#create em object
em_object=em.em_learner(em_params)
for iteration in range(em_params['num_iterations']):
	li_em_obj.append(em_object.e_step_m_step(tm,phi,y,iteration))# do one EM iteration
	li_w.append(utils.compute_exact_wass_loss(tm,em_object))
	print(iteration,"li_em_obj:",li_em_obj[-1],"li_w",li_w[-1])#print EM objective
	sys.stdout.flush()
	numpy.savetxt("log/w_loss-"+str(run_number)+"-"+\
				 str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
				 "-"+str(em_params['gaussian_variance'])+"-"+str(model_params['num_hidden_layers'])+"-"+str(model_params['lipschitz_constant'])+".txt",li_w)
	numpy.savetxt("log/em_loss-"+str(run_number)+"-"+\
				 str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
				 "-"+str(em_params['gaussian_variance'])+"-"+str(model_params['num_hidden_layers'])+"-"+str(model_params['lipschitz_constant'])+".txt",li_em_obj)
	numpy.savetxt("log/learned_probs-"+str(run_number)+"-"+\
				 str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
				 "-"+str(em_params['gaussian_variance'])+"-"+str(model_params['num_hidden_layers'])+"-"+str(model_params['lipschitz_constant'])+".txt",em_object.learned_priors)
	li_show=[]
	for index,m in enumerate(tm.models):
		mname="log/model-"+str(run_number)+"-"+\
				 str(model_params['num_samples'])+"-"+str(model_params['learning_rate'])+\
				 "-"+str(em_params['gaussian_variance'])+"-"+str(model_params['num_hidden_layers'])+\
				 "-"+str(model_params['lipschitz_constant'])+"-"+str(index)+".h5"
		m.save_weights(mname)




