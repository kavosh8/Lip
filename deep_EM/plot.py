import numpy
import matplotlib.pyplot
import matplotlib.pyplot as plt
import sys
li_k=[0.05,0.1,0.15,0.175,0.2,0.25,0.3,0.35,0.5,1.0]
li_w={}
li_em_obj={}
for variance in [0.05]:
	for k in li_k:
		temp_li=[]
		temp2_li=[]
		for run_ID in range(400):
			
			try:
				temp=numpy.loadtxt('log/w_loss-'+str(run_ID)+"-"+
					  str(k)+"-"+
				 	  str(variance)+".txt")
				not_exist=False
			except:
				not_exist=True
			if not_exist==True or type(temp)!=numpy.ndarray or temp.shape==() or len(temp)!=100:
				pass
			else:
				temp_li.append(temp)
				temp=numpy.loadtxt('log/em_obj-'+str(run_ID)+"-"+
					  str(k)+"-"+
				 	  str(variance)+".txt")
				#print(len(temp))
				temp2_li.append(temp)
		li_w[k]=temp_li		
		li_em_obj[k]=temp2_li
#sys.exit(1)
ax=plt.subplot(222)
for k in li_k:
	plt.plot(numpy.mean(li_w[k],axis=0),label="k="+str(k))
plt.xlabel('EM iterations',fontsize=14)
plt.ylabel('Wass \n loss',rotation=0,labelpad=20,fontsize=12)
plt.ylim([150,400])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_label_position("right")
plt.legend(fontsize=8)

ax=plt.subplot(221)
for k in li_k:
	plt.plot(-numpy.mean(li_em_obj[k],axis=0),label="k="+str(k))
plt.xlabel('EM iterations',fontsize=14)
plt.ylabel('negative\n log \n likelihood',rotation=0,labelpad=20,fontsize=12)
plt.ylim([0,7000])
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_label_position("left")
plt.legend(fontsize=8)

ax=plt.subplot(224)
y=[numpy.mean(li_w[x],axis=0)[-1] for x in li_k]
N=len(li_w[0.1])
n=numpy.sqrt(N)
yerr=[numpy.std(li_w[x],axis=0)[-1]/n for x in li_k]
plt.errorbar(li_k,y,yerr=yerr,lw=4)
plt.ylabel('final\n Wass \n loss',rotation=0,labelpad=20,fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_label_position("right")
#plt.ylim([170,270])
plt.xlabel('k',fontsize=14)

ax=plt.subplot(223)
y=[-numpy.mean(li_em_obj[x],axis=0)[-1] for x in li_k]
N=len(li_w[0.1])
n=numpy.sqrt(N)
yerr=[numpy.std(li_em_obj[x],axis=0)[-1]/n for x in li_k]
plt.errorbar(li_k,y,yerr=yerr,lw=4)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_label_position("left")
plt.xlabel('k',fontsize=14)
plt.ylabel('final\n negative\n log \n likelihood',rotation=0,labelpad=20,fontsize=12)
plt.tight_layout(pad=0.4, w_pad=1, h_pad=1)
plt.show()
#plt.legend()
#plt.show()
#print(li_w)
#plt.close()
