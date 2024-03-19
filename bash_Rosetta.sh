#!/bin/bash

#OAR -n Rosetta
#OAR -l /nodes=1/core=3,walltime=75:00:00  

# Change number of cores and walltime according to your run

# Virtual environment activation
source /data/tectonique/bouchary/virtualenv/bin/activate

# Run the model with mpi for parallelisation using python 3.11

# Command line if number of physical cores (line 4) = number of topographic profiles.
mpirun -np `cat $OAR_FILE_NODES|wc -l` --machinefile $OAR_NODE_FILE python3.11 Rosetta.py

# Command line if number of physical cores != number of topographic profiles. 
#mpirun -np <number of topo profiles> -oversubscribe --machinefile $OAR_NODE_FILE python3.11 Rosetta.py


# Run the script :
#   1- Log in : ssh -X login@ist-oar.u-ga.fr
#   2- Type the command :  oarsub -S --project iste-equ-<team> ./bash_Rosetta.sh
#   3- With notification : oarsub -S --project iste-equ-<team> ./bash_Rosetta.sh --notify "mail:<email adress>
#   4- Kill a job : oardel  <job_id>
#   5- Visualisation of cores utilisation : http://ist-oar.u-ga.fr/monika?Action=Display+nodes+for+these+properties&.cgifields=props

