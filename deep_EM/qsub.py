import os,sys,re,time
import os.path

bash_script = '''#!/bin/bash
#PBS -r n
#PBS -l nodes=4
#PBS -M kavosh.asadi8@gmail.com

source /home/kasadiat/.bashrc
export PYTHONPATH=/home/kasadiat/anaconda2/lib/python2.7/site-packages:/home/kasadiat/anaconda2
echo "prog started at: 'date'"
cd /home/kasadiat/Desktop/deep_EM

python main.py {} {} {}
'''


for run_number in range(400):
	for k in [0.05,0.1,0.15,0.175,0.2,0.25,0.3,0.35,0.5,1.0]:
		for variance in [0.1,0.05]:
			fname="log/w_loss-"+str(run_number)+"-"+str(k)+"-"+str(variance)+".txt"
			if os.path.isfile(fname)==False:
				outfile="pbs_files/EM_{}_{}_{}.pbs".format(str(run_number),str(k),str(variance))
				output=open(outfile, 'w')
				print >>output, (bash_script.format(str(run_number),str(k),str(variance)))
				output.close()
				cmd="qsub -l short %s" % outfile
				os.system(cmd)
				time.sleep(.2)
