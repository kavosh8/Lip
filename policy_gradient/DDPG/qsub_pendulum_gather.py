import os,sys,re,time

bash_script = '''#!/bin/bash
#PBS -r n
#PBS -l nodes=4
#PBS -M kavosh.asadi8@gmail.com

source /home/kasadiat/.bashrc
export PYTHONPATH=/home/kasadiat/anaconda2/lib/python2.7/site-packages:/home/kasadiat/anaconda2
echo "prog started at: 'date'"
cd /home/kasadiat/Desktop/Lipschitz/implementation/DDPG


python ddpg_main.py {} {} {}
echo "prog finished at: 'date'"
'''
env_name='Pendulum-v0'
env_test_name='Pendulum-v0'


for run_number in range(20):
	outfile="DDPG_{}_{}_{}.pbs".format(run_number,env_name,env_test_name)
	output=open(outfile, 'w')
	print >>output, (bash_script.format(run_number,env_name,env_test_name))
	output.close()
	cmd="qsub -l short %s" % outfile
	os.system(cmd)
	time.sleep(.1)
