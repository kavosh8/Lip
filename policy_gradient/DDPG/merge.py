li=['reps','reps_prime','actions']

for name in li:
	'''
	for line in open("log/"+l+"_"+str(0)+".csv", "r"):
	    fout.write(line)
	'''
	fout=open("offline_data/"+name+"_matrix.csv","a")
	for counter in range(3):
		with open("offline_data/"+name+"_"+str(counter)+".csv", "r") as f:
			l=0
			for line in f:
				l=l+1
				fout.write(line)
			print(counter,l)
			f.close()
	fout.close()