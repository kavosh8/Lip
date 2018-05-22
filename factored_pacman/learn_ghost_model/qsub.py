import os,sys,re,time
import os.path

bash_script = '''#!/bin/bash
#PBS -r n
#PBS -l nodes=4
#PBS -M kavosh.asadi8@gmail.com

source /home/kasadiat/.bashrc
export PYTHONPATH=/home/kasadiat/anaconda2/lib/python2.7/site-packages:/home/kasadiat/anaconda2
echo "prog started at: 'date'"
cd /home/kasadiat/Desktop/factored_pacman/learn_ghost_model/
python main.py {} {} {} {} {} {}
'''


for run_number in range(200):
	for num_samples in [5*49]:
		for learning_rate in [0.001]:
			for gaussian_variance in [0.05]:
				for num_hidden_layers in [2]:
					for lipschitz_constant in [0.1,0.15,0.2,0.25,0.3,0.5,0.75,1.0,2.0]:
						fname="log/w_loss-"+str(run_number)+"-"+str(num_samples)+"-"+str(learning_rate)+"-"+str(gaussian_variance)+"-"+str(num_hidden_layers)+"-"+str(lipschitz_constant)+".txt"
						if os.path.isfile(fname)==False:
							outfile="pbs_files/factored_pacman_{}_{}_{}_{}_{}_{}.pbs".format(str(run_number),str(num_samples),str(learning_rate),str(gaussian_variance),str(num_hidden_layers),str(lipschitz_constant))
							output=open(outfile, 'w')
							print >>output, (bash_script.format(str(run_number),str(num_samples),str(learning_rate),str(gaussian_variance),str(num_hidden_layers),str(lipschitz_constant)))
							output.close()
							cmd="qsub -l short %s" % outfile
							os.system(cmd)
							time.sleep(5)
