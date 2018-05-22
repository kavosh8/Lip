import numpy
import sys

class em_learner:
	def __init__(self, params):
		self.num_iterations=params['num_iterations']
		self.gaussian_variance=params['gaussian_variance']
		self.N=params['num_models']
		self.learned_priors=(self.N)*[1.0/self.N]

	def compute_posterior(self,tm,phi,y,iteration_number):
		o_li=tm.predict(phi)
		#print("o_li",o_li)
		y_arr = numpy.squeeze(numpy.asarray(y)).reshape(len(y),1)
		o_li_diff=[]
		for o in o_li:
			o_li_diff.append((o-y_arr).flatten())
		o_li_diff=numpy.transpose(numpy.array(o_li_diff))
		#print("o_li_diff",o_li_diff)
		#o_li_diff=-numpy.multiply(o_li_diff,o_li_diff)/gaussian_variance
		p_arr=numpy.zeros_like(o_li_diff)
		obj=0
		for i in range(o_li_diff.shape[0]):	
				#print(i,"o_li_diff",o_li_diff[i,:])
				mult=-numpy.multiply(o_li_diff[i,:],o_li_diff[i,:])/self.gaussian_variance
				#print(i,"mult",mult)
				#sys.exit(1)
				clipped=numpy.clip(mult,a_min=-600, a_max=600)
				#print(i,"clipped",clipped)
				exped=numpy.exp(clipped)
				top=numpy.multiply(exped,numpy.array(self.learned_priors))
				down=numpy.sum(top)
				
				#print(numpy.exp(shifted)/sum_val)
				p_arr[i,:]=top/down
				temp=numpy.multiply(p_arr[i,:],mult)+numpy.multiply(p_arr[i,:],numpy.log(self.learned_priors))
				#print(numpy.sum(temp))
				obj=obj+numpy.sum(temp)
				#print(i,p_arr[i,:])
				#sys.exit(1)
		p_arr=numpy.transpose(p_arr)
		#print(p_arr)
		p_li=[]
		for i in range(len(p_arr)):
			p_li.append(numpy.clip(p_arr[i,:],a_min=1e-20, a_max=1))
		#print("p_li",p_li)
		return p_li,obj
	def compute_learned_prior(self,w_li):
		for n in range(self.N):
			self.learned_priors[n]=numpy.mean(w_li[n])

	def e_step_m_step(self,tm,phi,y,iteration):
		w_li,obj=self.compute_posterior(tm,phi,y,iteration)#E step
		if iteration>0:
			tm.regression(phi,y,w_li)#M step
			self.compute_learned_prior(w_li)
		return obj