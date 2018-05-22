import numpy
import scipy
import scipy.stats as st



def TV(X,Y):
	tv=0.5*numpy.linalg.norm(numpy.array(X)-numpy.array(Y),ord=1)
	return tv

def KL(X,Y):
	out=0
	for i in range(len(X)):
		x,y=X[i,:],Y[i,:]
		out=out+st.entropy(x,qk=y)
	return out

def W(X,Y):
	out=0
	for i in range(len(X)):
		x,y=X[i,:],Y[i,:]
		out=out+my_ws(x, y)
	return out

def my_ws(X_in,Y_in):
	X=X_in[:]
	Y=Y_in[:]
	out=0
	movee=0
	for index in range(len(X)-1):
		#print(X,Y)
		movee=max(X[index],Y[index])-min(X[index],Y[index])
		out=out+movee
		if X[index]>Y[index]:
			X[index+1]=X[index+1]+movee
			X[index]=X[index]-movee
		elif Y[index]>X[index]:
			Y[index+1]=Y[index+1]+movee
			Y[index]=Y[index]-movee
	#print(X,Y)
	return out
