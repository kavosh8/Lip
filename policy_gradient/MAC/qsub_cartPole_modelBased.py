import os,sys,re,time

bash_script = '''#!/bin/bash
#PBS -r n
#PBS -l nodes=4
#PBS -M kavosh.asadi8@gmail.com

source /home/kasadiat/.bashrc
export PYTHONPATH=/home/kasadiat/anaconda2/lib/python2.7/site-packages:/home/kasadiat/anaconda2
echo "prog started at: 'date'"
cd /home/kasadiat/Desktop/Lipschitz/implementation/MAC


python run.py {} {} {}
echo "prog finished at: 'date'"
'''

for run in range(100):
	for k in [0.1,0.2,0.3,0.5,0.75,1.5,2]:
		environment='CartPolelearned'+str(k)+'-v0'
		environment_name_test='CartPole-v0'
		outfile="MAC_{}_{}_{}.pbs".format(str(run),environment,environment_name_test)
		output=open(outfile, 'w')
		print >>output, (bash_script.format(str(run),environment,environment_name_test))
		output.close()
		cmd="qsub -l short %s" % outfile
		os.system(cmd)
		time.sleep(.1)
