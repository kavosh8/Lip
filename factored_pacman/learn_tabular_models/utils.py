import numpy, scipy, sys, csv
'''
def wasserstein(true_probs,true_labels,estimated_probs,estimated_labels):
	W=scipy.stats.wasserstein_distance(true_labels, estimated_labels, u_weights=true_probs, v_weights=estimated_probs)
	return W

def load_data(fname):
	with open(fname, 'rb') as f:
		data = list(csv.reader(f))
	out=[]
	for d in data:
		temp=[]
		for s in d:
			temp.append(float(s))
		out.append(temp)
	return out

def load_synthetic_data(N):
	each=N/49
	li_s,li_sprime=[],[]
	for x in range(7):
		for y in range(7):
			s=[x,y]
			for _ in range(each):
				while True:
					case=numpy.random.randint(0,4)
					if case==0:
						s_p=[x+1,y]
					elif case==1:
						s_p=[x,y+1]
					elif case==2:
						s_p=[x-1,y]
					elif case==3:
						s_p=[x,y-1]
					if numpy.min(s_p)>=0 and numpy.max(s_p)<=6:
						break
				li_s.append(s)
				li_sprime.append(s_p)

	
	for t1,t2 in zip(li_s,li_sprime):
		print(t1,t2)
	sys.exit(1)
	
	return li_s,li_sprime

def create_matrices(li_samples,li_labels,model_params):
	N=len(li_samples)
	arr_samples=numpy.array(li_samples).reshape(N,model_params['observation_size'])
	#arr_samples=numpy.concatenate((arr_samples,numpy.ones(N).reshape(N,1)),axis=1)
	mat_samples=numpy.matrix(arr_samples)

	arr_labels=numpy.array(li_labels).reshape(N,model_params['observation_size'])
	mat_labels=numpy.matrix(arr_labels)
	return mat_samples,mat_labels
def compute_approx_wass_loss(tm,em_object):# only for 7*7 maze this one works should change to be robust to choice of maze size

	w_loss=0
	N=500
	for _ in range(N):
		sample_li=numpy.random.randint(0,7,2)
		estimated_labels=[m.predict(numpy.array(sample_li).reshape(1,2))[0] for m in tm.models]
		estimated_labels=numpy.array(estimated_labels)
		for i in range(estimated_labels.shape[1]):
			estimated_labels_column=estimated_labels[:,i].tolist()
			estimated_labels_probs=em_object.learned_priors
			#print(estimated_labels_column,estimated_labels_probs)
			if sample_li[i]>0 and sample_li[i]<6:
				true_labels=[sample_li[i]]*2+([sample_li[i]+1])*1+([sample_li[i]-1])*1
				true_probs=4*[(1./4)]
			elif sample_li[i]==0:
				true_labels=[sample_li[i]]*2+([sample_li[i]+1])*1+([sample_li[i]-1])*1
				true_probs=[(1./3)]*2+[(1./3)]*1+[0]*1
			elif sample_li[i]==6:
				true_labels=[sample_li[i]]*2+([sample_li[i]+1])*1+([sample_li[i]-1])*1
				true_probs=[(1./3)]*2+[0]*1+[(1./3)]*1
			w_loss=w_loss+wasserstein(true_probs,true_labels,estimated_labels_probs,estimated_labels_column)
	return w_loss/N
'''
def state_2_number(state):
	if len(state)!=2:
		print("wow")
		sys.exit(1)
	return state[0] + state[1]*7

def number_2_state(number):
	li=[]
	for _ in range(1):
		temp=int(number%7)
		li.append(temp)
		number=number/7
	li.append(number)
	return li