import numpy, metrics, sys
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import utils

N=10
num_experiments=50000
try:
	reward_type=sys.argv[1]
except:
	print("enter a reward design ... either structured or random")
	sys.exit(1)

W_correlation=[1.]
TV_correlation=[1.]
KL_correlation=[1.]
gamma_li=[0.05*x for x in range(1,20)]

for gamma in gamma_li:
	print("running {} experiments for gamma={}".format(num_experiments,gamma))

	li_TV,li_KL,li_W,li_planning_error=utils.experiment(num_experiments,gamma,N,reward_type)#run experiments and collect data
	TV_correlation.append(utils.compute_covariance(li_TV, li_planning_error))#compute covariances...
	KL_correlation.append(utils.compute_covariance(li_KL, li_planning_error))#..
	W_correlation.append(utils.compute_covariance(li_W, li_planning_error))#.

#printing stuff for the paper ....


plt.plot([0]+gamma_li,W_correlation,color='red',lw=4,label='Wasserstein')
plt.plot([0]+gamma_li,KL_correlation,color='blue',lw=4,label='KL')
plt.plot([0]+gamma_li,TV_correlation,color='green',lw=4,label='TV')
plt.ylabel('correlation',rotation=0,labelpad=-5,fontsize=14)
plt.ylim([0,1.1])
plt.xlabel(r'$\gamma$',fontsize=14)
ax=plt.gca()
ax.yaxis.set_label_position('right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(fontsize=14)
plt.show()
plt.close()

ax=plt.subplot(311)
plt.plot(li_W,li_planning_error,'o',color='red')
plt.xlabel("W",labelpad=-10,fontsize=14)
plt.xticks([0,2,4,7,10,12])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')



ax=plt.subplot(312)
plt.plot(li_KL,li_planning_error,'o',color='blue')
plt.xlabel("KL",labelpad=-10,fontsize=14)
ax.yaxis.set_label_position("right")
plt.xticks([0,1,3,6,8])
plt.ylabel("planning \n error",rotation=0,labelpad=20,fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')


ax=plt.subplot(313)
plt.plot(li_TV,li_planning_error,'o',color='green')
plt.xlabel("TV",labelpad=-10,fontsize=14)
plt.xticks([0,.1,.2,.3,.5,.6])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')


plt.show()
plt.close()

#printing stuff for the paper ....
