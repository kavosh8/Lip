import matplotlib.pyplot as plt
import numpy
x=[0.2,0.25,0.425,0.475,0.6,0.75,2]
y=[]
max_run=68
'''
plt.subplot(121)
for k in x:
	each_k=[]
	for run in range(max_run):
		temp=numpy.loadtxt('train_result_train_Pendulumlearned'+str(k)+'-v0_test_Pendulum-v0_run_'+str(run)+'.txt')
		each_k.append(temp)
	print(each_k)
	plt.plot(numpy.mean(each_k,axis=0),label=k)
	#y.append(numpy.mean(each_k))
#plt.plot(x,y)

plt.legend()
'''

plt.subplot(122)
for k in x:
	each_k=[]
	for run in range(max_run):
		temp=numpy.loadtxt('test_result_train_Pendulumlearned'+str(k)+'-v0_test_Pendulum-v0_run_'+str(run)+'.txt')
		each_k.append(temp)
	print(each_k)
	#plt.plot(numpy.mean(each_k,axis=0),label=k)
	y.append(numpy.mean(each_k))
plt.plot(x,y,lw=3)

plt.legend()
#plt.show()

plt.subplot(121)
for k in x:
	each_k=[]
	for run in range(max_run):
		temp=numpy.loadtxt('test_result_train_Pendulumlearned'+str(k)+'-v0_test_Pendulum-v0_run_'+str(run)+'.txt')
		each_k.append(temp)
	print(each_k)
	plt.plot(numpy.mean(each_k,axis=0),label=k)
	#y.append(numpy.mean(each_k))
#plt.plot(x,y)

plt.legend()
plt.show()

plt.close()
