import random
import numpy
import numpy.random
import sys
import numpy.linalg as lg
import matplotlib.pyplot as plt
import sys
import transition_model
import math
import em
import scipy.stats
import time
import tensorflow as tf
import utils

#make sure results are reproducable ...
try:
	run_ID=int(sys.argv[1])
	plot=False
except:
	run_ID=23
	plot=True

numpy.random.seed(run_ID)
random.seed(run_ID)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(run_ID)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#make sure results are reproducable ...


def plot_everything(li_samples,li_labels,tm,phi,ax):
	ax.clear()
	ax.plot(li_samples,li_labels,'o')
	y_li=tm.predict(phi)
	for index,y in enumerate(y_li):
		ax.plot(li_samples,y,'o',lw=2,label=index)
	plt.legend()

li_num_samples=5*[30]
num_lines=len(li_num_samples)
#get hyper-parameters as input
model_params={}
try:
	model_params['lipschitz_constant']=float(sys.argv[2])
except:
	model_params['lipschitz_constant']=.175
model_params['num_hidden_layers']=2
model_params['hidden_layer_nodes']=32
model_params['activation_fn']='relu'
model_params['learning_rate']=0.0001
model_params['observation_size']=1
model_params['num_models']=num_lines
model_params['num_epochs']=50
em_params={}
em_params['num_iterations']=100
try:
	em_params['gaussian_variance']=float(sys.argv[3])
except:
	em_params['gaussian_variance']=.05
	em_params['num_iterations']=200
#get hyper-parameters as input

em_params['num_models']=model_params['num_models']
if plot==True:
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
li_w=[]#list of wasserstein distance per iteration
li_em_obj=[]#list of em log-likelihood probability per iteration
#create data
li_samples,li_labels=utils.create_train_data(li_num_samples,num_lines)
phi,y=utils.create_matrices(li_samples,li_labels)
#create networks
tm=transition_model.neural_transition_model(model_params)
#create the em object
em_object=em.em_learner(em_params)

for iteration in range(em_params['num_iterations']):
	li_em_obj.append(em_object.e_step_m_step(tm,phi,y,iteration))#perform E step and M step
	if plot==True:
		plot_everything(li_samples,li_labels,tm,phi,ax)
		fig.savefig('save/visualize'+str(run_ID)+"-"+
				  str(model_params['lipschitz_constant'])+"-"+
			 	  str(em_params['gaussian_variance'])+'iteration-'+str(iteration)+'.pdf')
		if plot==True:
			if iteration==0:
				plt.pause(1)
			else:
				plt.pause(.5)
	li_w.append(utils.compute_wass_loss(tm,phi,em_object))
	print("li_w",li_w)
	print("li_em_obj",li_em_obj)
	sys.stdout.flush()
	numpy.savetxt("log/w_loss-"+str(run_ID)+"-"+
				  str(model_params['lipschitz_constant'])+"-"+
			 	  str(em_params['gaussian_variance'])+".txt",li_w)
	numpy.savetxt("log/em_obj-"+str(run_ID)+"-"+
				  str(model_params['lipschitz_constant'])+"-"+
			 	  str(em_params['gaussian_variance'])+".txt",li_em_obj)

