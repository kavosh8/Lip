import numpy
import scipy

def f0(x):
	return numpy.tanh(x)+3
def f1(x):
	return x*x
def f2(x):
	return numpy.sin(x)-5
def f3(x):
	return numpy.sin(x)-3
def f4(x):
	return numpy.sin(x)*numpy.sin(x)


def create_train_data(li_num_samples,num_lines):
	li_samples=[]
	li_labels=[]
	for l in range(num_lines):
		for n in range(li_num_samples[l]):
			sample=numpy.random.uniform(-2,2)
			if l==0:
				li_samples.append(sample)
				label=f0(sample)#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_labels.append(label)
					
			elif l==1:
				label=f1(sample)#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			elif l==2:
				label=f2(sample)#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			elif l==3:
				label=f3(sample)#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			elif l==4:
				label=f4(sample)#+numpy.random.normal(loc=0.0, scale=0.0,size=1)
				li_samples.append(sample)
				li_labels.append(label)
			else:
				print('not implemented yet ... aborting')
				sys.exit(1)
			#print(sample,label)
	return li_samples,li_labels

def wasserstein(true_probs,true_labels,estimated_probs,estimated_labels):
	W=scipy.stats.wasserstein_distance(true_labels, estimated_labels, u_weights=true_probs, v_weights=estimated_probs)
	return W

def compute_wass_loss(tm,phi,em_object):
	sample_li=numpy.arange(-2,2,0.01)
	total_wass=0
	for x in sample_li:
		true_labels=[f0(x),f1(x),f2(x),f3(x),f4(x)]
		true_probs=5*[0.2]
		estimated_labels=[m.predict(numpy.array(x).reshape(1,1))[0,0] for m in tm.models]
		estimated_probs=em_object.learned_priors
		total_wass=total_wass+wasserstein(true_probs,true_labels,estimated_probs,estimated_labels)
	#print("Wasserstein loss",total_wass)
	return total_wass
	#sys.exit(1)


def create_matrices(li_samples,li_labels):
	N=len(li_samples)
	arr_samples=numpy.array(li_samples).reshape(N,1)
	#arr_samples=numpy.concatenate((arr_samples,numpy.ones(N).reshape(N,1)),axis=1)
	mat_samples=numpy.matrix(arr_samples)

	arr_labels=numpy.array(li_labels).reshape(N,1)
	mat_labels=numpy.matrix(arr_labels)
	return mat_samples,mat_labels