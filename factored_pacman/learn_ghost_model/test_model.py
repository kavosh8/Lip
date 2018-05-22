import numpy
import matplotlib.pyplot as plt
import sys

li_k=[0.1,0.15,0.2,0.25,0.3,0.5,0.75,1.0]
li_w=[]
li_w_error=[]
li_em=[]
li_em_error=[]
for num_samples in [5*49]:
	for learning_rate in [0.001]:
		for gaussian_variance in [0.05]:
			for num_hidden_layers in [2]:
				for lipschitz_constant in li_k:
					li_temp=[]
					li_temp2=[]
					for run_number in range(20):
						temp=numpy.loadtxt('log/w_loss-'+str(run_number)+\
											'-'+str(num_samples)+'-'+str(learning_rate)+\
											'-'+str(gaussian_variance)+'-'+str(num_hidden_layers)+\
											'-'+str(lipschitz_constant)+'.txt')

						if type(temp)!=numpy.ndarray or temp.shape==() or len(temp)!=500:
							pass
						else:
							li_temp.append(temp[-1])
						temp2=numpy.loadtxt('log/em_loss-'+str(run_number)+\
											'-'+str(num_samples)+'-'+str(learning_rate)+\
											'-'+str(gaussian_variance)+'-'+str(num_hidden_layers)+\
											'-'+str(lipschitz_constant)+'.txt')
						
						if type(temp2)!=numpy.ndarray or temp2.shape==() or len(temp2)!=500:
							pass
						else:
							li_temp2.append(temp2[-1])
					print(lipschitz_constant,numpy.mean(li_temp))
					li_w.append(numpy.mean(li_temp))
					li_w_error.append(numpy.std(li_temp)/numpy.sqrt(len(li_temp)))
					li_em.append(-numpy.mean(li_temp2))
					li_em_error.append(-numpy.std(li_temp2)/numpy.sqrt(len(li_temp2)))

ax=plt.subplot(121)
plt.errorbar(x=li_k,y=li_w,yerr=li_w_error,lw=5)
plt.ylabel('final\n Wass \n loss',rotation=0,labelpad=20,fontsize=12)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_label_position("left")
plt.ylim([0.2,0.6])
plt.xlim([0.1,1])

#plt.ylim([170,270])
plt.xlabel('k',fontsize=14)
ax=plt.subplot(122)
plt.errorbar(x=li_k,y=li_em,yerr=li_em_error,lw=5)
plt.ylabel('final\n negative\n log \n likelihood',rotation=0,labelpad=20,fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_label_position("right")
plt.ylim([325,500])
plt.xlabel('k',fontsize=14)
plt.tight_layout(pad=0.4, w_pad=1, h_pad=1)
plt.show()