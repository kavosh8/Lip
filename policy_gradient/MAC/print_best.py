import numpy as np
import matplotlib.pyplot as plt
import sys
environment=sys.argv[1]
if environment=='CartPole-v0':
	num_trials=200
	y_max=200
	y_min=0
	num_episodes=1000
elif environment=='LunarLander-v2':
	num_trials=100
	y_max=250
	y_min=-400
	num_episodes=10000

li_ret=[]
num_avail=0
for run in range(num_trials):
	try:
		ret=np.loadtxt("best_results/"+environment+"/"+environment+"-best-"+str(run)+".txt")
		print(run,np.mean(ret))
		li_ret.append(ret)
		num_avail=num_avail+1
	except:
		print("best_results/"+environment+"/"+environment+"-best-"+str(run)+".txt","not found")
		#li_ret.append(1000*[-100])
mean=np.mean(li_ret[0:num_episodes],axis=0)
plt.plot(mean)
plt.ylim([y_min,y_max])
print(num_avail)
plt.show()
plt.close()