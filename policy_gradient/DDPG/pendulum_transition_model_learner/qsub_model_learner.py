import os,sys,re,time

bash_script = '''#!/bin/bash
#PBS -r n
#PBS -l nodes=4
#PBS -M kavosh.asadi8@gmail.com

source /home/kasadiat/.bashrc
export PYTHONPATH=/home/kasadiat/anaconda2/lib/python2.7/site-packages:/home/kasadiat/anaconda2
echo "prog started at: 'date'"
cd /home/kasadiat/Desktop/Lipschitz/implementation/DDPG/pendulum_transition_model_learner


python learner.py {} {}
echo "prog finished at: 'date'"
'''


for k in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.425,0.45,0.475,0.5,0.55,0.6,0.75,1,2,5]:
	for run in range(20):
		outfile="model_learner_{}_{}.pbs".format(k,run)
		output=open(outfile, 'w')
		print >>output, (bash_script.format(k,run))
		output.close()
		cmd="qsub -l short %s" % outfile
		os.system(cmd)
		time.sleep(.1)
