import numpy
import metrics

def get_normalized_matrix(N):
	
	T=numpy.random.random((N*N)).reshape(N,N)
	for i in range(N):
		T[i,:]=T[i,:]/numpy.sum(T[i,:])
	return T

def compute_planning_error(T,T_hat,R,gamma,N):
	I=numpy.identity(N)
	V=numpy.linalg.inv(I-gamma*T).dot(R)
	V_hat=numpy.linalg.inv(I-gamma*T_hat).dot(R)
	error=numpy.linalg.norm(V-V_hat,ord=2)
	return error

def experiment(num_experiments,gamma,N,reward_type):
	li_TV,li_KL,li_W=[],[],[]
	li_planning_error=[]
	for experiment in range(num_experiments):
		numpy.random.seed(experiment) #choose a seed for reproducability
		T,T_hat=get_normalized_matrix(N),get_normalized_matrix(N)
		
		if reward_type=='random':
			R=numpy.random.random(N).reshape(N,1)
		elif reward_type=='structured':
			R=numpy.array([x for x in range(N)]).reshape(N,1)
		else:
			print("undefined reward structure ...")
			sys.exit(1)

		li_planning_error.append(compute_planning_error(T,T_hat,R,gamma,N))
		li_TV.append(metrics.TV(T,T_hat)),li_KL.append(metrics.KL(T,T_hat)),li_W.append(metrics.W(T,T_hat))
	return li_TV,li_KL,li_W,li_planning_error

def compute_covariance(li_1,li_2):
	return numpy.cov(li_1, li_2)[0,1]/numpy.sqrt(numpy.var(li_1)*numpy.var(li_2))