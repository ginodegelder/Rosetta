=============
Rosetta model
=============
-----------------------------------------------
Resolving sea-level, coral growth and tectonics
-----------------------------------------------

Description
===========

This model resolves blabla

Installation 
------------

Environment required : Linux or Mac
 
You might need to create a virtual environment to load all the modules needed for the model.

ISTerre OAR server log in::
        
        ssh -X <login>@ist-oar.u-ga.fr
        password : <your password>
        login@ist-oar:$ cd /data/tectonique/login

Creating a python 3.11 virtual environment named 'virtualenv' in 'login' directory::

        login@ist-oar:/data/tectonique/login$ python3.11 -m venv virtualenv

Activation::

        login@ist-oar:/data/tectonique/login$ source virtualenv/bin/activate

You are now in your python3.11 virtual environment. 
To install the needed libraries (make sure your virtual env is activated)::

        pip install -r requirements.txt

to quit the virtual env::
        
        deactivate

Initialisation :
----------------

You should only modify "Inputs.py" and "bash_Rosetta.sh" if you just want to run the model. 
To access source code of the reef construction, everything is in the folder "Library/reef". 
To access source code of Bayesian MCMC inversion, it's in "Library/mcmc".

Everything is detailed in "Inputs.py". Just follow the indicated format for the dictionnaries.

The files with the observed topographies have to be of the following form (x and y separated by tab) :

   - x(m): Distance in meters
   - y(m): Elevation in meters
   - Do not write the header, only topo values

+------+------+
| x(m) | y(m) |
+======+======+
| 0    | -40  |
+------+------+
| ...  | ...  |
+------+------+
| 1000 | 0    |
+------+------+

Put every observed topography files, in a .dat format, in the Topo_obs directory. If you want to use another directory or path, don't forget to change the path in "Inputs.py".

After changing "Inputs.py", modify the number of cores and walltime (time submitted to the server) in the bash file "bash_Rosetta.sh".

 - If you can use one core for each profile, the bash file should contain the following lines::
        
        #!/bin/bash
        #OAR -n Rosetta
        #OAR -l /nodes=1/core=<number of topo profiles>,walltime=<time needed in HH:MN:SS format>  

        source <path to the directory containing the virtual environment>/virtualenv/bin/activate

        mpirun -np `cat $OAR_FILE_NODES|wc -l` --machinefile $OAR_NODE_FILE python3.11 Rosetta.py


 - If the number of cores available and the number of topographic profiles you want to model are different, you need to modify the bash file as follows::

        #!/bin/bash
        #OAR -n Rosetta
        #OAR -l /nodes=1/core=<number of available cores>,walltime=<time needed in HH:MN:SS format>  

        source <path to the directory containing the virtual environment>/virtualenv/bin/activate

        mpirun -np <number of topo profiles> -oversubscribe --machinefile $OAR_NODE_FILE python3.11 Rosetta.py


You can also modify line 3 as::

	#OAR -n <the name you want to give to your run>

Run the model:
--------------

Make sure to deactivate your virtual environment, the bash file will activate it automatically.

The model runs on passive job using the bash file "bash_Rosetta.sh". After modifiyng it type the following command::
	
        oarsub -S --project iste-equ-<team> ./bash_Rosetta.sh --notify "mail:<email adress>" -t besteffort

- "oarsub -S" : asks to run a script on the oar server, using the commands in a bash file.
- "--project iste-equ-<team>" : tells to the server on which team you are. Modify <team> by your team name.
- "./bash_Rosetta.sh" : asks to use the bash file "bash_Rosetta.sh". Works if you type the command in the folder containing this file, or modify the path.
- "--notify "mail:<email adress>"" : optional, asks to send an email to <email adress> when the model is finished or killed.
- "-t besteffort" : optional, asks the server to run your script on any free cores. Usefull for tests, better to remove for long walltimes.

The time needed for the inversion varies a lot with the time length of one simulation (older value in sea_level) but also with a wide platform ("init__lterr") and gentle initial slope ("grid__slopi"). Try with approximately 10 to a 1000 simulations (n_samples) to see how much time is needed for 1 simulation.

The time needed for one simulation decreases bit by bit for long runs due to cache effect. It might divide the time for one simulation by 2 or 3, depending on n_samples and the time length for one simulation. 

Outputs :
---------

All the outputs are in "Outs/FigS4d" folder.
The outputs from one model are stored in a directory named : "Figs_<n_samples>_sig.<sigma>_ip.<ipstep>_<DAY-MONTH-YEAR_HOUR-MIN at the end of the run>".
 - There will be one subfolder for each profile, named by the name of the profile, containing the histogramms for each reef parameters corresponding to the profile, its best, mean and median profile in .txt and the profile figures. 
 - "Dataframes" folder contains all the data saved at the end of the inversion. Data are saved in .pkl format, readable with panda.
 - "SL" folder contains histogramms of the free SL nodes, best, mean and median SL history in .txt format, and SL plots.
 - "Stats" folder contains the evolution of the best acceptance ratio of the profiles, the best loglikelihood, the acceptance ratio for each parameters, and the step evolution (prop_S) for each free parameters. 
 - "AA-Inputs" file contains all the Inputs for the model.
 - "BestLogLike" contains the value of the best loglike reached during the inversion.

Example of an output folder in "Outs/FigS4d"

::

    Figs_10_sig.10_ip.150_19-02-2024_17.01
    ├── Dataframes
    │    ├── df_SL_6.0ky.pkl
    │    ├── My_topo_file_1_param1-param2.pkl
    │    ├── [other files for topo_file_1]
    │    ├── My_topo_file_2_param1-param2.pkl
    │    └── [other files for topo_file_2]
    ├── SL
    │    ├── BestSL.txt
    │    ├── Histogram-6.0ky.png
    │    ├── MeanSL.txt
    │    ├── MedianSL.txt
    │    ├── Sea-Level.pdf
    │    └── Sea-Level_median_percentiles.pdf
    ├── Stats
    │    ├── Accept_ratio.png
    │    ├── Loglikelihood.png
    │    ├── Param_accept_ratio.png
    │    └── Prop_S.png
    ├── My_topo_file_1
    │    ├── BestProfile.txt
    │    ├── [2D Histograms for topo 1 free reef params]
    │    ├── MeanProfile.txt
    │    ├── MedianProfile.txt
    │    ├── Profile_median_percentiles.pdf
    │    └── Profiles.pdf
    ├── My_topo_file_2
    │    ├── BestProfile.txt
    │    ├── [2D Histograms for topo 1 free reef params]
    │    ├── MeanProfile.txt
    │    ├── MedianProfile.txt
    │    ├── Profile_median_percentiles.pdf
    │    └── Profiles.pdf
