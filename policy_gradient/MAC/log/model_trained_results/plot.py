import numpy
import matplotlib.pyplot as plt
'''
plt.subplot(311)
max_runs=50
for k in [0.1,0.2,0.3,0.5,0.75,1.5,2]:
	li_k=[]
	for run in range(100):
		temp=numpy.loadtxt('returns_train_CartPolelearned'+str(k)+'-v0_test_CartPole-v0_'+str(run)+'_on_train.txt')
		li_k.append(temp)
	temp=numpy.mean(li_k,axis=0)
	plt.plot(temp,label=str(k))
plt.legend()
'''
plt.subplot(121)
for k in [0.1,0.2,0.3,0.5,0.75,1.5,2]:
	li_k=[]
	for run in range(100):
		temp=numpy.loadtxt('returns_train_CartPolelearned'+str(k)+'-v0_test_CartPole-v0_'+str(run)+'_on_test.txt')
		print(len(temp))
		li_k.append(temp)
	temp=numpy.mean(li_k,axis=0)
	plt.plot(temp,label=str(k))
plt.legend()

plt.subplot(122)
li_all=[]
for k in [0.1,0.2,0.3,0.5,0.75,1.5,2]:
	li_k=[]
	for run in range(100):
		try:
			temp=numpy.loadtxt('returns_train_CartPolelearned'+str(k)+'-v0_test_CartPole-v0_'+str(run)+'_on_test.txt')
			li_k.append(temp)
		except:
			pass
	print(numpy.mean(li_k))
	li_all.append(numpy.mean(li_k))
plt.plot([0.1,0.2,0.3,0.5,0.75,1.5,2],li_all,lw=4)
plt.xticks([0.2,0.5,0.75,1.5,2])
plt.yticks([75,190,100,150])
plt.ylim([50,150])
plt.legend()
plt.show()


plt.close()