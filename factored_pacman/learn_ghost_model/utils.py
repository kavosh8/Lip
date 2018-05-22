import numpy, scipy, sys, csv
from scipy.optimize import linprog
import decimal

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

def state_2_number(state):
	print(state)
	if len(state)!=2:
		print("wow")
		sys.exit(1)
	return state[0] + state[1]*7

def number_2_state(number):
	li=[]
	for _ in range(2):
		temp=int(number%7)
		li.append(temp)
		number=number/7
	li.append(number)
	return li

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
	return li_s,li_sprime

def create_matrices(li_samples,li_labels,model_params):
	N=len(li_samples)
	arr_samples=numpy.array(li_samples).reshape(N,model_params['observation_size'])
	#arr_samples=numpy.concatenate((arr_samples,numpy.ones(N).reshape(N,1)),axis=1)
	mat_samples=numpy.matrix(arr_samples)

	arr_labels=numpy.array(li_labels).reshape(N,model_params['observation_size'])
	mat_labels=numpy.matrix(arr_labels)
	return mat_samples,mat_labels

def compute_exact_wass_loss(tm,em_object):
	temp=0
	for x in range(7):
		for y in range(7):
			estimated_labels,estimated_labels_probs=[m.predict(numpy.array([x,y]).reshape(1,2))[0].tolist() for m in tm.models],em_object.learned_priors
			true_labels,true_probs=compute_true_labels_and_probs(x,y)
			temp=temp+create_and_solve_LP(estimated_labels,estimated_labels_probs,true_labels,true_probs)
	return temp/(7*7)

def compute_true_labels_and_probs(x,y):
	li=[]
	if x>0:
		li.append([x-1,y])
	if x<6:
		li.append([x+1,y])
	if y>0:
		li.append([x,y-1])
	if y<6:
		li.append([x,y+1])
	return li, len(li)*[(1./len(li))]

def create_and_solve_LP(estimated_labels,estimated_labels_probs,true_labels,true_probs):
	L1,L2=len(estimated_labels),len(true_labels)
	#compute the cost for each x,y pair
	D=numpy.zeros((L1,L2))
	for i in range(L1):
		for j in range(L2):
			D[i][j]=numpy.linalg.norm(x=numpy.array(estimated_labels[i])-numpy.array(true_labels[j]))
	D=D.reshape((L1*L2))
	#compute the cost for each x,y pair
	#Ax=b
	A=numpy.zeros((L1+L2,L1*L2))
	for i in range(L1):#equality constraints for marginal of x
		for j in range(L2):
			A[i,i*L2+j]=1
	for j in range(L2):#equality constraints for marginals of y
		for i in range(L1):
			A[L1+j,j+L2*i]=1
	
	b=numpy.zeros(L1+L2)
	for i in range(L1):
		b[i]=numpy.around(estimated_labels_probs[i],5)
	for j in range(L2):
		b[L1+j]=numpy.around(true_probs[j],5)
	temp=numpy.sum(b[0:L1])-numpy.sum(b[L1:])
	b[-1]=b[-1]+temp
	#Ax=b

	A_ub=numpy.zeros((L1*L2,L1*L2))#make sure each p_xy is non-negative
	for i in range(L1*L2):
		A_ub[i,i]=-1
	b_ub=numpy.zeros(L1*L2)

	opt_res = linprog(D,A_ub=A_ub, b_ub=b_ub, A_eq=A, b_eq=b)
	if opt_res.status==2:
		print("LP solver messed up!")
		print(A)
		print(b)
		print(b)
		print(D)
		sys.exit(1)
			
	return opt_res.fun

def compute_approx_wass_loss(tm,em_object):# only for 7*7 maze this one works should change to be robust to choice of maze size

	w_loss=0
	N=500
	for _ in range(N):
		sample_li=numpy.random.randint(0,7,2)

		estimated_labels=[m.predict(numpy.array(sample_li).reshape(1,2))[0].tolist() for m in tm.models]
		estimated_labels_probs=em_object.learned_priors
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