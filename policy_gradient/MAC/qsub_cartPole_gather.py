import os,sys,re,time

bash_script = '''#!/bin/bash
#PBS -r n
#PBS -l nodes=4
#PBS -M kavosh.asadi8@gmail.com

source /home/kasadiat/.bashrc
export PYTHONPATH=/home/kasadiat/anaconda2/lib/python2.7/site-packages:/home/kasadiat/anaconda2
echo "prog started at: 'date'"
cd /home/kasadiat/Desktop/Lipschitz/implementation/MAC


python run.py {}
echo "prog finished at: 'date'"
'''
environment='cartPole'


for run_number in range(20):
	outfile="MAC_{}.pbs".format(environment+"-"+str(run_number))
	output=open(outfile, 'w')
	print >>output, (bash_script.format(str(run_number)))
	output.close()
	cmd="qsub -l short %s" % outfile
	os.system(cmd)
	time.sleep(1)
