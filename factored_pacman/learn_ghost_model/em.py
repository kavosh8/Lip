import numpy
import sys
import numpy.linalg

class em_learner:
	"""docstring for ClassName"""
	def __init__(self, params):
		self.num_iterations=params['num_iterations']
		self.gaussian_variance=params['gaussian_variance']
		self.N=params['num_models']
		self.learned_priors=(self.N)*[1.0/self.N]
		self.observation_size=params['observation_size']

	def log_likelihood(self,sample_labels,line_labels):
		line_likelihoods=[]
		for o in line_labels:# for each fn output
			temp=[]
			for o_each,y_each in zip(o,sample_labels):# for each sample y, fn y_hat
				nm=numpy.linalg.norm(o_each-y_each)
				temp.append(-(nm*nm)/(2.*self.gaussian_variance))# log and e cancel out, since we assume a gaussian
			line_likelihoods.append(temp)
		line_likelihoods=numpy.transpose(numpy.array(line_likelihoods))#convert list of log likelihoods to a 2D array
		#of size number of samples * number of fn s
		return line_likelihoods

	def posterior(self,line_likelihoods):
		p_arr=numpy.zeros_like(line_likelihoods)
		obj=0
		num_samples=line_likelihoods.shape[0]
		for i in range(num_samples):
			#compute probabilities for each sample
			clipped=numpy.clip(line_likelihoods[i,:],a_min=-600, a_max=600)
			exped=numpy.exp(clipped)
			top=numpy.multiply(exped,numpy.array(self.learned_priors))
			p_arr[i,:]=top/numpy.sum(top)
			#compute probabilities for each sample
			#compute em objective note that E[log xy] is E[log x] + E[log y]
			obj=obj+numpy.dot(p_arr[i,:],line_likelihoods[i,:])+numpy.dot(p_arr[i,:],numpy.log(self.learned_priors))
			#compute em objective
		p_arr=numpy.transpose(p_arr)
		p_li=[]
		for i in range(len(p_arr)):
			p_li.append(numpy.clip(p_arr[i,:],a_min=1e-3, a_max=1))
		return p_li,obj

	def compute_learned_prior(self,w_li):# compute best priors, where best is defined as the prior that maximizes lower bound
		for n in range(self.N):#in this case \mean_x p(z|x)
			self.learned_priors[n]=numpy.mean(w_li[n])

	def m_step(self,tm,phi,y,w_li,iteration):
		tm.regression(phi,y,w_li,iteration)#M step fits however many functions (fn)
		self.compute_learned_prior(w_li)# and the prior to maximize the lower bound.

	def e_step(self,tm,phi,y,iteration_number):
		line_labels=tm.predict(phi)# get y_hat from each fn, given x values. a list.
		sample_labels = numpy.squeeze(numpy.asarray(y)).reshape(len(y),self.observation_size)
		line_likelihoods=self.log_likelihood(sample_labels,line_labels)# get log likelihood of each (x,y) coming from each fn
		p_li,obj=self.posterior(line_likelihoods,)#get posterior p(z|(x,y_hat))
		return p_li,obj

	def e_step_m_step(self,tm,phi,y,iteration):
		w_li,obj=self.e_step(tm,phi,y,iteration)#E step computes posteriors w_li and EM objective obj
		self.m_step(tm,phi,y,w_li,iteration)
		return obj