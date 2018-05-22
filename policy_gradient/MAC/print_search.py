import numpy as np
import matplotlib.pyplot as plt
import sys
environment=sys.argv[1]
if environment=='CartPole-v0':
	repeat=20
	num_runs=9
	max_timesteps=1000
if environment=='LunarLander-v2':
	repeat=8
	num_runs=9
	max_timesteps=20000

def smooth(li):
	li_smoothed=[]
	for i in range(len(li)):
		min_index=max(0,i-20)
		max_index=min(max_timesteps,i+20)
		li_smoothed.append(np.mean(li[min_index:max_index]))
	return li_smoothed

li_ret=[]
for run in range(num_runs):
	li_set=[]
	num_available=0
	for rep in range(repeat):
		try:
			ret=np.loadtxt("search_results/"+environment+"/"+environment+"-"+str(run*repeat+rep)+".txt")
			li_set.append(ret)
			num_available=num_available+1
		except:
			print("search_results/"+environment+"/"+environment+"-"+str(run*repeat+rep)+".txt")
			#li_set.append(2000*[-100])
	if len(li_set)==0:
		print("setting number:",run,"mean return:",-100,"number of trials:",num_available)
		li_ret.append(max_timesteps*[-100])
	else:
		print("setting number:",run,"mean return:",np.mean(li_set),"number of trials:",num_available)
		li_ret.append(np.mean(li_set,axis=0))

li_ending=[x[-1000:] for x in li_ret]

mean_ret=np.mean(li_ret,axis=1)
order=np.argsort(mean_ret)
mean_ret=np.mean(li_ending,axis=1)
order=np.argsort(mean_ret)
print(order)


if sys.argv[2]=='plot':
	for index in order[-2:]:
		print(index)
		smoothed=smooth(li_ret[index])
		plt.plot(smoothed)
plt.xlim([0,max_timesteps])
plt.show()
plt.close()

