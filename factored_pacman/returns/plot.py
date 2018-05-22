import numpy
import matplotlib.pyplot as plt

x=[0.1,0.15,0.2,0.25,0.3,0.5,0.75,1.0]
y=[]
y_std=[]
ax = plt.subplot(111)
N=200
for lipschitz_constant in x:
	li=[]
	for run_number in range(N):
		for learning_rate in [0.001]:
			for gaussian_variance in [0.05]:
				try:
					temp=numpy.loadtxt('stochastic/'+\
						str(lipschitz_constant)+"-"+str(run_number)+"-"+str(learning_rate)+"-"+str(gaussian_variance)+'.txt')
					#print
					if len(temp==100):
						li.append(numpy.mean(temp))
				except:
					print(lipschitz_constant)
					pass
	y.append(-numpy.mean(li))
	print(lipschitz_constant,-numpy.mean(li))
	y_std.append(numpy.std(li)/numpy.sqrt(N))
ax.errorbar(x,y,yerr=y_std,lw=4,color='red')

tab=[]
for j in range(10):
	temp=numpy.loadtxt('tabular/'+str(j)+'.txt')
	tab.append(numpy.mean(temp))
ax.plot(x,len(x)*[-numpy.mean(tab)],'--',label='tabular baseline',lw=4)
#print(numpy.mean(tabular_result))
'''
random_result=numpy.loadtxt('other/random.txt')[0]
plt.plot(x,len(x)*[-random_result])
'''
det_result=numpy.loadtxt('other/deterministic.txt')
#print(numpy.mean(det_result))
ax.plot(x,len(x)*[-numpy.mean(det_result)],'--',label='deterministic baseline',lw=4)
plt.xticks([0.1,0.2,0.3,0.5,1])
plt.yticks([-14,-11,-9,-6])
#plt.xlim([0.15,1.1])
plt.xlabel("k",size=14)
plt.ylabel("average \n return",rotation=0,size=14)
ax.yaxis.set_label_position("right")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.legend()
#plt.yscale('log')
plt.show()
plt.close()
