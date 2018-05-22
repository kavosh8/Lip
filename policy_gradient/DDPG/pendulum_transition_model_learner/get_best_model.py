import numpy
import matplotlib.pyplot as plt
from shutil import copyfile
x=[0.2,0.25,0.3,0.35,0.425,0.45,0.475,0.6,0.75,2.0]
y=[]
for k in x:
	each_k=[]
	for run in range(20):
		temp=numpy.loadtxt('loss_k'+str(k)+"_"+str(run)+".txt")
		each_k.append(temp)
	#print(k,numpy.min(each_k))
	#print(k,numpy.median(each_k))
	#print(k,numpy.sort(each_k)[9])
	median_index=numpy.argsort(each_k)[9]
	print(k,median_index)
	y.append(numpy.sort(each_k)[9])
	src='pendulum_learned_k'+str(k)+'_'+str(median_index)+'.h5'
	dst='median_models/pendulum_learned_k'+str(k)+'.h5'
	copyfile(src, dst)

plt.plot(x,y)
plt.show()